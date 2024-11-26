import os
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


cont_min_ = [0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
cont_max_ = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
cont_diff_ = [cont_max_[i] - cont_min_[i] for i in range(len(cont_min_))]
continuous_range_ = range(1, 14)
categorical_range_ = range(14, 40)

if __name__ == "__main__":
    data_filepath = '.'
    save_filepath = '.'
    file_name = 'train.txt'

    raw_data = []
    with tqdm(total = os.path.getsize(f"{data_filepath}/{file_name}")) as pbar:
        with open(f"{data_filepath}/{file_name}") as fp:
            for line in fp:
                raw_data.append(line)
                pbar.update(len(line))
    print(f"Number of samples in {file_name} : {len(raw_data)}")

    target = np.zeros((len(raw_data), ), np.float32)
    continuous_data = np.zeros((len(raw_data), len(continuous_range_)), np.float32)
    categorical_data = np.zeros((len(raw_data), len(categorical_range_)), np.uint64)

    for sample_idx, raw_sample in enumerate(tqdm(raw_data)):
        features = raw_sample.rstrip('\n').split('\t')

        target[sample_idx] = float(features[0])

        for idx in continuous_range_:
            if features[idx] != "":
                continuous_data[sample_idx][idx - 1] = (float(features[idx]) - cont_min_[idx - 1]) / \
                    cont_diff_[idx - 1]

        for idx in categorical_range_:
            if features[idx] == "":
                categorical_data[sample_idx][idx - len(continuous_range_) - 1] = hash(str(idx)) % np.iinfo(np.uint64).max
            else:
                categorical_data[sample_idx][idx - len(continuous_range_) - 1] = int(features[idx], 16)

    target_train, target_test, continuous_data_train, continuous_data_test, \
        categorical_data_train, categorical_data_test = train_test_split(target, continuous_data, \
        categorical_data, test_size=0.2, random_state=42)

    CATEGORICAL_COLUMNS = ['C' + str(i) for i in range(1, 27)]
    np.savez(
        os.path.join(save_filepath, "train"),
        target=target_train,
        continuous_data=continuous_data_train,
        categorical_data=categorical_data_train,
        categorical_columns=CATEGORICAL_COLUMNS,
    )
    print("save train.npz done")

    np.savez(
        os.path.join(save_filepath, "test"),
        target=target_test,
        continuous_data=continuous_data_test,
        categorical_data=categorical_data_test,
        categorical_columns=CATEGORICAL_COLUMNS,
    )
    print("save test.npz done")
