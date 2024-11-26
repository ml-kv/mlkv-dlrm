## Operating system and software dependency requirements
Ubuntu (20.04 LTS x86/64) packages
```
sudo apt-get update
sudo apt-get install -y curl git build-essential python3-dev python3-pip
```

Python packages
```
conda create -n MLKV-DLRM python=3.10
conda activate MLKV-DLRM
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pandas scikit-learn tqdm
```

[Rust](https://www.rust-lang.org/tools/install)
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain=1.70.0 -y
source "$HOME/.cargo/env"
rustc --version
rustup self uninstall
```

[NATS](https://docs.nats.io/running-a-nats-service/introduction/installation)
```
curl -L https://github.com/nats-io/nats-server/releases/download/v2.9.15/nats-server-v2.9.15-linux-amd64.zip -o nats-server.zip
unzip nats-server.zip -d nats-server
sudo cp nats-server/nats-server-v2.9.15-linux-amd64/nats-server /usr/bin/
```

## PERSIA
Build
```
git clone -b persia git@github.com:ml-kv/mlkv-dlrm.git persia
cd persia
USE_CUDA=1 NATIVE=1 pip3 install .
pip uninstall persia
```

Test
```
cd examples/src/adult-income/data && ./prepare_data.sh
cd ../
export PATH=/home/$USER/.local/bin/:$PATH
honcho start -e .honcho.env
```

Benchmark
```
git clone -b main git@github.com:ml-kv/mlkv-dlrm.git mlkv-dlrm
cd criteo-ad/data
./download.sh
python3 data_preprocess.py
cd ../
honcho start -e .honcho.env
```

## PERSIA with MLKV
Build
```
sudo apt-get install make cmake clang
sudo apt-get install uuid-dev libaio-dev libtbb-dev libgflags-dev
pip uninstall cmake
git clone -b persia-mlkv git@github.com:ml-kv/mlkv-dlrm.git persia-mlkv
cd persia-mlkv/rust
git clone -b main git@github.com:ml-kv/mlkv-rust.git mlkv-rust
rm -rf mlkv-rust/.git
cd ../
USE_CUDA=1 NATIVE=1 pip3 install .
pip uninstall persia
```

Benchmark
```
git clone -b main git@github.com:ml-kv/mlkv-dlrm.git mlkv-dlrm
cd criteo-ad/data
./download.sh
python3 data_preprocess.py
cd ../
honcho start -e .honcho.env

cd criteo-terabyte/data
nohup ./download.sh &
nohup python3 categorical_data_preprocess.py &
nohup python3 categorical_data_json.py &
python3 data_preprocess.py
cd ../
honcho start -e .honcho.env
```

## PERSIA with FASTER
Build
```
sudo apt-get install make cmake clang
sudo apt-get install uuid-dev libaio-dev libtbb-dev libgflags-dev
pip uninstall cmake
git clone -b persia-faster git@github.com:ml-kv/mlkv-dlrm.git persia-faster
cd persia-faster/rust
git clone -b rust-faster git@github.com:ml-kv/mlkv-rust.git rust-faster
rm -rf rust-faster/.git
cd ../
USE_CUDA=1 NATIVE=1 pip3 install .
pip uninstall persia
```
