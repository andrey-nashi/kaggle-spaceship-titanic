

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from ds_base import  AbstractDataset


class TabularDataset(AbstractDataset):


    FEATURE_UNUSED = -1
    FEATURE_FLOAT = 0
    FEATURE_INT = 1
    FEATURE_CATEGORICAL = 2
    FEATURE_BOOLEAN = 3

    LABEL_TYPE_DIRECT = 0
    LABLE_TYPE_ONEHOT = 1

    KEY_FEATURES = "features"
    KEY_LABEL = "label"

    SCALER_STANDARD = 0
    SCALER_MINMAX = 1

    def __init__(self):
        super(TabularDataset, self).__init__()
        self.header = None
        self.categories = {}

    def load_from_csv(self, path_csv: str, delimiter: str = ",", value_types: list = None,
                      label_index: int = -1, label_type: int = LABEL_TYPE_DIRECT,
                      label_max_value: int = 1, label_min_value: int = 0):

        f = open(path_csv, "r")
        header = f.readline()
        self.header = header.strip().split(delimiter)

        if label_index == -1:
            label_index = len(self.header) - 1
        if value_types is None:
            value_types = [self.FEATURE_FLOAT] * len(self.header)

        for line in f:
            shards = line.strip().split(delimiter)

            fv = []

            counter = 0
            for i in range(0, len(shards)):
                feature_name = self.header[i]
                feature_value = shards[i]
                feature_type = value_types[i]

                if feature_type == self.FEATURE_FLOAT:
                    feature_value_new = self._convert_str2float(feature_name, feature_value)
                    fv.append(feature_value_new)
                elif feature_type == self.FEATURE_INT:
                    feature_value_new = self._convert_str2int(feature_name, feature_value)
                    fv.append(feature_value_new)
                elif feature_type == self.FEATURE_BOOLEAN:
                    feature_value_new = self._convert_str2boolean(feature_name, feature_value)
                    fv.append(feature_value_new)
                elif feature_type == self.FEATURE_CATEGORICAL:
                    feature_value_new = self._convert_str2categorical(feature_name, feature_value)
                    fv.append(feature_value_new)
                else:
                    if i < label_index:
                        counter += 1

            label_index_up = label_index - counter
            label = fv.pop(label_index_up)

            sample = {}

            if label_type == self.LABEL_TYPE_DIRECT:
                sample[self.KEY_FEATURES] = fv
                sample[self.KEY_LABEL] = label
            elif label_type == self.LABLE_TYPE_ONEHOT:
                dim = label_max_value - label_min_value
                label_norm = label - label_min_value
                label_norm = self._convert_int2onehot(label_norm, dim)

                sample[self.KEY_FEATURES] = fv
                sample[self.KEY_LABEL] = label_norm

            print(sample)
            self.samples_table.append(sample)

        f.close()


    def scale_feature_vectors(self, scaler_type: int = SCALER_STANDARD):
        if scaler_type == self.SCALER_STANDARD:
            scaler = StandardScaler()
        elif scaler_type == self.SCALER_MINMAX:
            scaler = MinMaxScaler()
        else:
            return

        temp_features = [f[self.KEY_FEATURES] for f in self.samples_table]
        temp_features = np.array(temp_features)
        temp_features = scaler.fit_transform(np.array(temp_features))

        for i in range(0, len(self.samples_table)):
            self.samples_table[i][self.KEY_FEATURES] = temp_features[i].tolist()


    def _convert_str2float(self, name: str, value: any) -> float:
        if len(value) == 0: return -1
        return float(value)

    def _convert_str2int(self, name: str, value: any) -> int:
        if len(value) == 0: return -1
        return int(float(value))

    def _convert_str2boolean(self, name: str, value: any) -> int:
        if len(value) == 0: return -1
        if value == "True" or value == "true":
            return 1
        else:
            return 0

    def _convert_str2categorical(self, name: str, value: any) -> int:
        if len(value) == 0: return -1
        if name not in self.categories:
            self.categories[name] = []
        if value not in self.categories[name]:
            self.categories[name].append(value)

        index = self.categories[name].index(value)
        return index

    def _convert_int2onehot(self, label: int, max_dim: int) -> list:
        x = [0] * max_dim
        x[label] = 1
        return x

    def __len__(self):
        return len(self.samples_table)

    def __getitem__(self, sample_index: int):
        sample = self.samples_table[sample_index]
        feature_vector = sample[self.KEY_FEATURES]
        label_vector = sample[self.KEY_LABEL]

        return feature_vector, label_vector



d = TabularDataset()
header_out = [
    TabularDataset.FEATURE_UNUSED,
    TabularDataset.FEATURE_CATEGORICAL,
    TabularDataset.FEATURE_BOOLEAN,
    TabularDataset.FEATURE_CATEGORICAL,
    TabularDataset.FEATURE_INT,
    TabularDataset.FEATURE_CATEGORICAL,
    TabularDataset.FEATURE_CATEGORICAL,
    TabularDataset.FEATURE_INT,
    TabularDataset.FEATURE_BOOLEAN,
    TabularDataset.FEATURE_INT,
    TabularDataset.FEATURE_INT,
    TabularDataset.FEATURE_INT,
    TabularDataset.FEATURE_INT,
    TabularDataset.FEATURE_INT,
    TabularDataset.FEATURE_CATEGORICAL,
    TabularDataset.FEATURE_CATEGORICAL,
    TabularDataset.FEATURE_BOOLEAN,
]
d.load_from_csv("../data/train-clean.csv", value_types=header_out)

print(d.__getitem__(6))
d.scale_feature_vectors()
print(d.__getitem__(6))