import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler



class TabularDataset:

    # ---- Feature types
    FEATURE_FLOAT = 0
    FEATURE_INT = 1
    FEATURE_CATEGORICAL = 2
    FEATURE_BOOLEAN = 3

    # ---- Policies for handling NAN data
    NAN2DATA_MIN = 0
    NAN2DATA_MAX = 1
    NAN2DATA_DISTRIBUTION = 2
    NAN2DATA_ZERO = 3

    # ---- Scaling methods
    SCALER_STANDARD = 0
    SCALER_MINMAX = 1


    def __init__(self):
        super(TabularDataset, self).__init__()
        # ---- Column names
        self.tbl_col_name_features = []
        self.tbl_col_name_label = None
        # ---- Column types as defined by this class
        self.tbl_col_types = {}
        # ---- Table of label -> count
        self.tbl_label_categories = None

        self.df = None

    def _nan2data_distribution(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        distribution = df[column_name].value_counts(normalize=True)
        new_val = np.random.choice(distribution.index, size=df[column_name].isna().sum(), p=distribution.values)
        df.loc[df[df[column_name].isna()].index, column_name] = new_val
        return df

    def _nan2data_min(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        if self.tbl_col_types in [self.FEATURE_CATEGORICAL, self.FEATURE_BOOLEAN]:
            values = dict(df[column_name].value_counts())
            new_val = min(values, key=values.get)
        else:
            new_val = min(df[column_name])
        df.loc[df[df[column_name].isna()].index, column_name] = new_val
        return df

    def _nan2data_max(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        if self.tbl_col_types in [self.FEATURE_CATEGORICAL, self.FEATURE_BOOLEAN]:
            values = dict(df[column_name].value_counts())
            new_val = max(values, key=values.get)
        else:
            new_val = max(df[column_name])
        df.loc[df[df[column_name].isna()].index, column_name] = new_val
        return df

    def _nan2data_zero(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        df.loc[df[df[column_name].isna()].index, column_name] = 0
        return df

    def _handle_nan(self, col_nan_policy: dict):
        for column_name in col_nan_policy:

            # ----------------------------------------
            if self.tbl_col_types[column_name] in [self.FEATURE_CATEGORICAL, self.FEATURE_BOOLEAN]:
                policy = col_nan_policy[column_name]
                if policy == self.NAN2DATA_DISTRIBUTION:
                    self.df = self._nan2data_distribution(self.df, column_name)
                elif policy == self.NAN2DATA_MIN:
                    self.df = self._nan2data_min(self.df, column_name)
                elif policy == self.NAN2DATA_MAX:
                    self.df = self._nan2data_max(self.df, column_name)

            # ----------------------------------------
            if  self.tbl_col_types[column_name] in [self.FEATURE_INT, self.FEATURE_FLOAT]:
                policy = col_nan_policy[column_name]

                if policy == self.NAN2DATA_DISTRIBUTION:
                    self.df = self._nan2data_distribution(self.df, column_name)
                elif policy == self.NAN2DATA_MIN:
                    self.df = self._nan2data_min(self.df, column_name)
                elif policy == self.NAN2DATA_MAX:
                    self.df = self._nan2data_max(self.df, column_name)
                elif policy == self.NAN2DATA_ZERO:
                    self.df = self._nan2data_zero(self.df, column_name)

    def load_from_csv(self, path_csv: str, col_name_label: str, col_types: dict, col_nan_policy: dict = None):
        df = pd.read_csv(path_csv)

        # ---- Load and extract categories, set types
        for column_name in col_types:
            self.tbl_col_name_features.append(column_name)
            self.tbl_col_types[column_name]  = col_types[column_name]

        self.tbl_col_name_label = col_name_label
        self.tbl_label_categories = dict(df[col_name_label].value_counts())
        self.df = df

        if col_nan_policy is not None:
            self._handle_nan(col_nan_policy)

    def get_table_info(self):
        output = {
            "col_name_features": self.tbl_col_name_features,
            "col_name_labels": self.tbl_col_name_label,
            "column_types": self.tbl_col_types,
            "labels": self.tbl_label_categories,
            "total_count": int(self.df.shape[0])
        }

        return output

    def get_feature_distribution(self, column_name: str):
        distribution = self.df[column_name].value_counts(normalize=True)
        distribution = dict(distribution)
        return distribution

    def compute_feature_label_distribution(self, feature_col_name: str, is_normalize: bool = False):
        """
        For the given column, if it is of categorical/boolean type, for each individual
        unique feature value and label combination count how many vectors are there
        with this combination.
        :param feature_col_name: name of the column
        :return: dict (or None if not categorical)
        dict[category_name][label] -> number of feature vectors
        """
        output = {}

        feature_col_type = self.tbl_col_types[feature_col_name]
        label_col_name = self.tbl_col_name_label
        if feature_col_type not in [self.FEATURE_CATEGORICAL, self.FEATURE_BOOLEAN]:
            return None

        df = self.df

        temp_table = df.groupby([feature_col_name, label_col_name])[label_col_name].size()
        if is_normalize:
            temp_norm = df.groupby([feature_col_name])[label_col_name].size()
            temp_table = temp_table / temp_norm
        temp_table = dict(temp_table)

        for key in temp_table:
            category = key[0]
            label = key[1]
            if category not in output:
                output[category] = {}
            output[category][label] = temp_table[key]

        return output

    def compute_feature_histogram(self, feature_col_name: str, bins=10):
        """
        Compute value/feature-vector count histogram for the given column
        :param feature_col_name: name of the column
        :param bins: number of bins if column is of FLOAT type
        :return: dict[col_name] -> value
        """
        feature_col_type = self.tbl_col_types[feature_col_name]
        if feature_col_type in [self.FEATURE_CATEGORICAL, self.FEATURE_BOOLEAN]:
            return dict(self.df[feature_col_name].value_counts())
        if feature_col_type == self.FEATURE_INT:
            temp = dict(self.df[feature_col_name].value_counts())
            output = {int(k):temp[k] for k in temp}
            return output
        if feature_col_type == self.FEATURE_FLOAT:
            hist, bins = np.histogram(self.df[feature_col_name], bins=bins)
            output = {}
            for bin_value, hist_value in zip(bins, hist):
                output[float(bin_value)] = hist_value
            return output

    def alter_nan2data(self, column_name: str, policy: int, arg: any = None):

        if policy == self.NAN2DATA_DISTRIBUTION:
            new_val = np.random.choice(list(arg.keys()), size=self.df[column_name].isna().sum(), p=list(arg.values()))
            self.df.loc[self.df[self.df[column_name].isna()].index, column_name] = new_val
        elif policy in [self.NAN2DATA_MIN, self.NAN2DATA_MAX]:
            self.df.loc[self.df[self.df[column_name].isna()].index, column_name] = arg
        else:
            self.df.loc[self.df[self.df[column_name].isna()].index, column_name] = 0










    def alter_scale_features(self, scaler_type: int = SCALER_STANDARD):
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





    def __len__(self):
        return len(self.samples_table)

    def __getitem__(self, sample_index: int):
        sample = self.samples_table[sample_index]
        feature_vector = sample[self.KEY_FEATURES]
        label_vector = sample[self.KEY_LABEL]

        return feature_vector, label_vector
