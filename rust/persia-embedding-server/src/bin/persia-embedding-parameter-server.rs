#![allow(clippy::needless_return)]

#[macro_use]
extern crate shadow_rs;

use std::{path::PathBuf, sync::Arc, ffi::CString, fs::File, io::Read};

use persia_libs::{anyhow::Result, color_eyre, hyper, rand, rand::Rng, serde::{self, Deserialize}, tracing, tracing_subscriber};
use structopt::StructOpt;

use persia_common::utils::start_deadlock_detection_thread;
use persia_embedding_config::{
    EmbeddingConfig, EmbeddingParameterServerConfig, PerisaJobType, PersiaCommonConfig,
    PersiaGlobalConfig,
};
use persia_embedding_server::embedding_parameter_service::{
    EmbeddingParameterNatsService, EmbeddingParameterNatsServiceResponder,
    EmbeddingParameterService, EmbeddingParameterServiceInner,
};
use persia_incremental_update_manager::PerisaIncrementalUpdateManager;
use persia_model_manager::EmbeddingModelManager;

#[derive(Debug, StructOpt, Clone)]
#[structopt()]
struct Cli {
    #[structopt(long)]
    port: u16,
    #[structopt(long)]
    replica_index: usize,
    #[structopt(long)]
    replica_size: usize,
    #[structopt(long, env = "PERSIA_GLOBAL_CONFIG")]
    global_config: PathBuf,
    #[structopt(long, env = "PERSIA_EMBEDDING_CONFIG")]
    embedding_config: PathBuf,
}

fn load_boringml_from_json(store: Arc<mlkv_rust::FasterKv>) {
    #[derive(Deserialize, Debug)]
    #[serde(crate = "self::serde")]
    struct CategoricalData {
        C1: Vec<u64>, C2: Vec<u64>, C3: Vec<u64>, C4: Vec<u64>, C5: Vec<u64>,
        C6: Vec<u64>, C7: Vec<u64>, C8: Vec<u64>, C9: Vec<u64>, C10: Vec<u64>,
        C11: Vec<u64>, C12: Vec<u64>, C13: Vec<u64>, C14: Vec<u64>, C15: Vec<u64>,
        C16: Vec<u64>, C17: Vec<u64>, C18: Vec<u64>, C19: Vec<u64>, C20: Vec<u64>,
        C21: Vec<u64>, C22: Vec<u64>, C23: Vec<u64>, C24: Vec<u64>, C25: Vec<u64>, C26: Vec<u64>,
    }

    let mut file = File::open("/mnt/nvme1n1/categorical_data.json").unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();
    let categorical_data: CategoricalData = serde_json::from_str(&contents).unwrap();

    let shared_categorical_data = vec![
        std::sync::Arc::new(categorical_data.C1), std::sync::Arc::new(categorical_data.C2), std::sync::Arc::new(categorical_data.C3),
        std::sync::Arc::new(categorical_data.C4), std::sync::Arc::new(categorical_data.C5), std::sync::Arc::new(categorical_data.C6),
        std::sync::Arc::new(categorical_data.C7), std::sync::Arc::new(categorical_data.C8), std::sync::Arc::new(categorical_data.C9),
        std::sync::Arc::new(categorical_data.C10), std::sync::Arc::new(categorical_data.C11), std::sync::Arc::new(categorical_data.C12),
        std::sync::Arc::new(categorical_data.C13), std::sync::Arc::new(categorical_data.C14), std::sync::Arc::new(categorical_data.C15),
        std::sync::Arc::new(categorical_data.C16), std::sync::Arc::new(categorical_data.C17), std::sync::Arc::new(categorical_data.C18),
        std::sync::Arc::new(categorical_data.C19), std::sync::Arc::new(categorical_data.C20), std::sync::Arc::new(categorical_data.C21),
        std::sync::Arc::new(categorical_data.C22), std::sync::Arc::new(categorical_data.C23), std::sync::Arc::new(categorical_data.C24),
        std::sync::Arc::new(categorical_data.C25), std::sync::Arc::new(categorical_data.C26)
    ];

    let feature_index_prefix_dict: Vec<u64> = vec![
        /*C1*/ 4503599627370496, /*C2*/ 9007199254740992, /*C3*/ 13510798882111488, /*C4*/ 18014398509481984, /*C5*/ 22517998136852480,
        /*C6*/ 27021597764222976, /*C7*/ 31525197391593472, /*C8*/ 36028797018963968, /*C9*/ 40532396646334464, /*C10*/ 45035996273704960,
        /*C11*/ 49539595901075456, /*C12*/ 54043195528445952, /*C13*/ 58546795155816448, /*C14*/ 63050394783186944, /*C15*/ 67553994410557440,
        /*C16*/ 72057594037927936, /*C17*/ 76561193665298432, /*C18*/ 81064793292668928, /*C19*/ 85568392920039424, /*C20*/ 90071992547409920,
        /*C21*/ 94575592174780416, /*C22*/ 99079191802150912, /*C23*/ 103582791429521408, /*C24*/ 108086391056891904, /*C25*/ 112589990684262400,
        /*C26*/ 117093590311632896
    ];

    let num_threads = 32;
    let feature_spacing = (1u64 << (u64::BITS - 12 as u32)) - 1;
    let feature_dim = 16;

    for column_idx in 0..26 {
        println!("C{:?} : {:?}", column_idx, shared_categorical_data[column_idx].len());

        let handles: Vec<std::thread::JoinHandle<_>> = (0..num_threads)
            .map(|thread_id| {
                let store = store.clone();
                let shared_categorical_data_column = shared_categorical_data[column_idx].clone();

                let start_idx = shared_categorical_data_column.len() / num_threads * thread_id;
                let end_idx = shared_categorical_data_column.len()  / num_threads * (thread_id + 1);
                let index_prefix = feature_index_prefix_dict[column_idx];
                std::thread::spawn(move || {
                    let mut rng = rand::thread_rng();
                    let mut value = vec![0 as f32; feature_dim * 3];

                    store.start_session();
                    for i in start_idx..end_idx {
                        let cur_key: u64 = shared_categorical_data_column[i] % feature_spacing + index_prefix;
                        for j in 0..feature_dim {
                            value[j] = rng.gen_range(-0.01..0.01) as f32;
                        }

                        let upsert_status = store.upsert(cur_key, value.as_mut_ptr() as *mut u8,
                            (std::mem::size_of::<f32>() * feature_dim * 3) as u64);
                        assert!(upsert_status == mlkv_rust::faster_status::FasterStatus::OK as u8);
                        if i % 1000000 == 0 {
                            store.complete_pending(true);
                        }
                    }
                    store.stop_session();
                })
            }).collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install().unwrap();
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_env("LOG_LEVEL"))
        .init();

    shadow!(build);
    eprintln!("project_name: {}", build::PROJECT_NAME);
    eprintln!("is_debug: {}", shadow_rs::is_debug());
    eprintln!("version: {}", build::version());
    eprintln!("tag: {}", build::TAG);
    eprintln!("commit_hash: {}", build::COMMIT_HASH);
    eprintln!("commit_date: {}", build::COMMIT_DATE);
    eprintln!("build_os: {}", build::BUILD_OS);
    eprintln!("rust_version: {}", build::RUST_VERSION);
    eprintln!("build_time: {}", build::BUILD_TIME);
    let args: Cli = Cli::from_args();

    start_deadlock_detection_thread();

    PersiaGlobalConfig::set_configures(
        &args.global_config,
        args.port,
        args.replica_index,
        args.replica_size,
    )?;

    EmbeddingConfig::set(&args.embedding_config)?;

    let embedding_config = EmbeddingConfig::get()?;
    let common_config = PersiaCommonConfig::get()?;
    let server_config = EmbeddingParameterServerConfig::get()?;
    let inc_update_manager = PerisaIncrementalUpdateManager::get()?;
    let embedding_model_manager = EmbeddingModelManager::get()?;
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let filename = CString::new("/mnt/nvme1n1/testdb").unwrap();
    let store = Arc::new(mlkv_rust::FasterKv::new(128 * 1024 * 1024, 64 * 1024 * 1024 * 1024, filename));
    //load_boringml_from_json(store.clone());

    let inner = Arc::new(EmbeddingParameterServiceInner::new(
        server_config,
        common_config,
        embedding_config,
        inc_update_manager,
        embedding_model_manager,
        args.replica_index,
        store,
    ));

    let service = EmbeddingParameterService {
        inner: inner.clone(),
        shutdown_channel: Arc::new(persia_libs::async_lock::RwLock::new(Some(tx))),
    };

    let server = hyper::Server::bind(&([0, 0, 0, 0], args.port).into())
        .tcp_nodelay(true)
        .serve(hyper::service::make_service_fn(|_| {
            let service = service.clone();
            async move { Ok::<_, hyper::Error>(service) }
        }));

    let job_type = &inner.get_job_type()?;
    let _responder = match job_type {
        PerisaJobType::Infer => None,
        _ => {
            let nats_service = EmbeddingParameterNatsService {
                inner: inner.clone(),
            };
            let responder = EmbeddingParameterNatsServiceResponder::new(nats_service).await;
            Some(responder)
        }
    };

    match job_type {
        PerisaJobType::Infer => {
            let common_config = PersiaCommonConfig::get()?;
            let embedding_cpk = common_config.infer_config.embedding_checkpoint.clone();
            inner.load(embedding_cpk).await?;
        }
        _ => {}
    }
    let graceful = server.with_graceful_shutdown(async {
        rx.await.ok();
    });

    if let Err(err) = graceful.await {
        tracing::error!("embedding server exited with error: {:?}!", err);
    } else {
        tracing::info!("embedding server exited successfully");
    }

    Ok(())
}
