import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from .ds_base import  AbstractDataset

class TabularDataset(AbstractDataset):

    # ---- Feature conversion
    FEATURE_FLOAT = 0
    FEATURE_INT = 1
    FEATURE_CATEGORICAL = 2
    FEATURE_BOOLEAN = 3

    # ---- Label conversion
    LABEL_TYPE_DIRECT = 0
    LABLE_TYPE_ONEHOT = 1

    KEY_FEATURES = "features"
    KEY_LABEL = "label"

    SCALER_STANDARD = 0
    SCALER_MINMAX = 1

    NAN2DATA_MIN = 0
    NAN2DATA_MAX = 1
    NAN2DATA_DISTRIBUTION = 2
    NAN2DATA_ZERO = 3

    def __init__(self):
        super(TabularDataset, self).__init__()

        # ---- Column names
        self.tbl_column_names = []
        self.tbl_label_name = None
        # ---- Column types as defined by this class
        self.tbl_column_types = {}
        # ---- Table of col_name -> {category: count} for original data
        self.tbl_column_categories_og = {}
        self.tbl_column_categories = {}
        self.tbl_category_encoder = {}
        # ---- Table of label -> count
        self.tbl_label_categories = None



    def _nan2data_distribution(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        distribution = df[column_name].value_counts(normalize=True)
        new_val = np.random.choice(distribution.index, size=df[column_name].isna().sum(), p=distribution.values)
        df.loc[df[df[column_name].isna()].index, column_name] = new_val
        return df

    def _nan2data_min(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        if self.tbl_column_types in [self.FEATURE_CATEGORICAL, self.FEATURE_BOOLEAN]:
            values = dict(df[column_name].value_counts())
            new_val = min(values, key=values.get)
        else:
            new_val = min(df[column_name])
        df.loc[df[df[column_name].isna()].index, column_name] = new_val
        return df

    def _nan2data_max(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        if self.tbl_column_types in [self.FEATURE_CATEGORICAL, self.FEATURE_BOOLEAN]:
            values = dict(df[column_name].value_counts())
            new_val = max(values, key=values.get)
        else:
            new_val = max(df[column_name])
        df.loc[df[df[column_name].isna()].index, column_name] = new_val
        return df

    def _nan2data_zero(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        df.loc[df[df[column_name].isna()].index, column_name] = 0
        return df

    def load_from_csv(self, path_csv: str, col_name_label: str, col_names_cvrt: dict = None, col_names_policy: dict = None):
        df = pd.read_csv(path_csv)

        # ---- Load and extract categories, set types
        for column_name in col_names_cvrt:
            self.tbl_column_names.append(column_name)
            self.tbl_column_types[column_name]  = col_names_cvrt[column_name]
            if col_names_cvrt[column_name] in [self.FEATURE_CATEGORICAL, self.FEATURE_BOOLEAN]:
                self.tbl_column_categories_og[column_name] = dict(df[column_name].value_counts())

        self.tbl_label_name = col_name_label
        self.tbl_label_categories = dict(df[col_name_label].value_counts())

        # ---- Handle missing values according to given policies
        for column_name in col_names_cvrt:

            # ----------------------------------------
            if col_names_cvrt[column_name] in [self.FEATURE_CATEGORICAL, self.FEATURE_BOOLEAN]:
                policy = col_names_policy[column_name]
                if policy == self.NAN2DATA_DISTRIBUTION:
                    df = self._nan2data_distribution(df, column_name)
                elif policy == self.NAN2DATA_MIN:
                    df = self._nan2data_min(df, column_name)
                elif policy == self.NAN2DATA_MAX:
                    df = self._nan2data_max(df, column_name)
                self.tbl_column_categories[column_name] = dict(df[column_name].value_counts())

            # ----------------------------------------
            if  col_names_cvrt[column_name] in [self.FEATURE_INT, self.FEATURE_FLOAT]:
                policy = col_names_policy[column_name]

                if policy == self.NAN2DATA_DISTRIBUTION:
                    df = self._nan2data_distribution(df, column_name)
                elif policy == self.NAN2DATA_MIN:
                    df = self._nan2data_min(df, column_name)
                elif policy == self.NAN2DATA_MAX:
                    df = self._nan2data_max(df, column_name)
                elif policy == self.NAN2DATA_ZERO:
                    df = self._nan2data_zero(df, column_name)

        # ---- Compute some extra statistics




        # ---- Convert the dataset



    def get_eda_info(self):
        output = {
            "column_names": self.tbl_column_names,
            "label_name": self.tbl_label_name,
            "column_types": self.tbl_column_types,
            "column_categories_og": self.tbl_column_categories_og,
            "column_categories_up": self.tbl_column_categories,
            "label_categories": self.tbl_label_categories
        }

        return output

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

    def get_all_features(self) -> np.ndarray:
        output = []
        for sample in self.samples_table:
            output.append(sample[self.KEY_FEATURES])
        return np.array(output)


    def get_all_labels(self) -> np.ndarray:
        output = []
        for sample in self.samples_table:
            output.append(sample[self.KEY_LABEL])
        return np.array(output)


    def _convert_str2float(self, name: str, value: any) -> float:
        if len(value) == 0:
            return None
        return float(value)

    def _convert_str2int(self, name: str, value: any) -> int:
        if len(value) == 0:
            return None
        return int(float(value))

    def _convert_str2boolean(self, name: str, value: any) -> int:
        if len(value) == 0:
            return None
        if value == "True" or value == "true":
            return 1
        else:
            return 0

    def _convert_str2categorical(self, name: str, value: any) -> int:
        if len(value) == 0: return -1
        if name not in self.category_table:
            self.category_table[name] = []
        if value not in self.category_table[name]:
            self.category_table[name].append(value)

        index = self.category_table[name].index(value)
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

    def get_item_og(self, sample_index: int):
        output = self.samples_origin[sample_index]
        return output