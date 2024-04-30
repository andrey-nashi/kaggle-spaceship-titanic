from sklearn.model_selection import train_test_split
from tqdm import tqdm

from core.ds_tabular import *
from core.ml_legacy import *


def write_predictions(model, dataset, path_output):
    progress_bar = tqdm(total=len(dataset))
    f = open(path_output, "w")
    f.write("PassengerId,Transported\n")
    for i in range(len(dataset)):
        progress_bar.update(1)
        sample = dataset.__getitem__(i)
        sample_origin = dataset.get_item_og(i)

        fv = sample[0]
        label = model.predict([fv])[0]
        f.write(str(sample_origin[0]) + "," + str(bool(label)) + "\n")
    f.close()


path_train = "./data/train-clean.csv"
path_test = "./data/test-clean.csv"
path_output = "./data/submission-01.csv"



col_names_cvrt = {
    "HomePlanet": TabularDataset.FEATURE_CATEGORICAL,
    "CryoSleep": TabularDataset.FEATURE_BOOLEAN,
    "Cabin-1": TabularDataset.FEATURE_CATEGORICAL,
    "Cabin-2": TabularDataset.FEATURE_INT,
    "Cabin-3": TabularDataset.FEATURE_CATEGORICAL,
    "Destination": TabularDataset.FEATURE_CATEGORICAL,
    "Age": TabularDataset.FEATURE_FLOAT,
    "VIP": TabularDataset.FEATURE_BOOLEAN,
    "RoomService": TabularDataset.FEATURE_FLOAT,
    "FoodCourt": TabularDataset.FEATURE_FLOAT,
    "ShoppingMall": TabularDataset.FEATURE_FLOAT,
    "Spa": TabularDataset.FEATURE_FLOAT,
    "VRDeck": TabularDataset.FEATURE_FLOAT,
    "Transported": TabularDataset.FEATURE_BOOLEAN,
}

col_names_policy = {
    "HomePlanet": TabularDataset.NAN2DATA_DISTRIBUTION,
    "CryoSleep": TabularDataset.NAN2DATA_DISTRIBUTION,
    "Cabin-1": TabularDataset.NAN2DATA_DISTRIBUTION,
    "Cabin-2": TabularDataset.NAN2DATA_DISTRIBUTION,
    "Cabin-3": TabularDataset.NAN2DATA_DISTRIBUTION,
    "Destination": TabularDataset.NAN2DATA_DISTRIBUTION,
    "Age": TabularDataset.NAN2DATA_DISTRIBUTION,
    "VIP": TabularDataset.NAN2DATA_DISTRIBUTION,
    "RoomService": TabularDataset.NAN2DATA_ZERO,
    "FoodCourt": TabularDataset.NAN2DATA_ZERO,
    "ShoppingMall": TabularDataset.NAN2DATA_ZERO,
    "Spa": TabularDataset.NAN2DATA_ZERO,
    "VRDeck": TabularDataset.NAN2DATA_ZERO,
    "Transported": TabularDataset.NAN2DATA_DISTRIBUTION
}

col_name_label = "Transported"


dataset_train = TabularDataset()
dataset_train.load_from_csv(path_train, col_name_label, col_names_cvrt, col_names_policy)
eda = dataset_train.get_eda_info()

for k in eda:
    print(k, eda[k])

"""
col_names_cvrt = {
    "f_1": TabularDataset.FEATURE_FLOAT,
    "f_2": TabularDataset.FEATURE_FLOAT,
    "f_3": TabularDataset.FEATURE_FLOAT,
    "f_4": TabularDataset.FEATURE_FLOAT,
    "f_5": TabularDataset.FEATURE_FLOAT,
    "label": TabularDataset.FEATURE_INT,
}

col_names_policy = {
    "f_1": TabularDataset.NAN2DATA_ZERO,
    "f_2": TabularDataset.NAN2DATA_ZERO,
    "f_3": TabularDataset.NAN2DATA_ZERO,
    "f_4": TabularDataset.NAN2DATA_ZERO,
    "f_5": TabularDataset.NAN2DATA_ZERO,
    "label": TabularDataset.NAN2DATA_ZERO,
}

col_name_label = "label"

dataset_train = TabularDataset()
dataset_train.load_from_csv("./data/train_data.csv", col_name_label, col_names_cvrt, col_names_policy)
eda = dataset_train.get_eda_info()

for k in eda:
    print(k, eda[k])
"""
"""
dataset_test = TabularDataset()
dataset_test.load_from_csv(path_test, column_types=header_out, is_skip_nan=False)

features = dataset_train.get_all_features()
labels = dataset_train.get_all_labels()

print(features)
plot_pca_3d(features, labels)

model = mll_train_svm(features, labels, kernel="rbf")

path_output = "./data/submission-svm-rbf-mm.csv"
write_predictions(model, dataset_test, path_output)
"""

