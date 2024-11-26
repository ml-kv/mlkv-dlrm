import os
import numpy as np
from tqdm import tqdm


continuous_range_ = range(1, 14)
categorical_range_ = range(14, 40)

if __name__ == "__main__":
    data_filepath = '.'
    save_filepath = '.'
    file_name = 'day'

    categorical_data = [set() for _ in categorical_range_]
    for i in range(24):
        raw_data = []
        with tqdm(total = os.path.getsize(f"{data_filepath}/{file_name}_{i}")) as pbar:
            with open(f"{data_filepath}/{file_name}_{i}") as fp:
                for line in fp:
                    raw_data.append(line)
                    pbar.update(len(line))

        for sample_idx, raw_sample in enumerate(tqdm(raw_data)):
            features = raw_sample.rstrip('\n').split('\t')

            for idx in categorical_range_:
                categorical_data[idx - len(continuous_range_) - 1].add(features[idx])

    CATEGORICAL_COLUMNS = ['C' + str(i) for i in range(1, 27)]
    for column_idx, column_name in enumerate(CATEGORICAL_COLUMNS):
        print(column_name, ": ", len(categorical_data[column_idx]))

    np.savez(
        os.path.join(save_filepath, "categorical_data"),
        categorical_data=categorical_data,
        categorical_columns=CATEGORICAL_COLUMNS,
    )
    print("save categorical_data.npz done")
