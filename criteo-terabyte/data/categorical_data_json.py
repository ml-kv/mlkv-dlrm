import numpy as np
import json
from tqdm import tqdm


if __name__ == "__main__":
    data_filepath = "./categorical_data.npz"
    with np.load(data_filepath, allow_pickle=True) as data:
        categorical_data = data["categorical_data"]
        categorical_columns = data["categorical_columns"]

    feat_ids = {}

    for column_idx, column_name in enumerate(categorical_columns):
        feat_ids[column_name] = []

        for categorical_feature in tqdm(categorical_data[column_idx]):
            if categorical_feature == "":
                categorical_feature = hash(str(column_idx + 14)) % np.iinfo(np.uint64).max
            else:
                categorical_feature = int(categorical_feature, 16)

            feat_ids[column_name].append(categorical_feature)
    
        print(column_name, " : ", len(feat_ids[column_name]))

    with open("categorical_data.json", "w") as outfile:
        json.dump(feat_ids, outfile)
