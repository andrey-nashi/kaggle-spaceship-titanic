import argparse

from core.ds_tabular import *
from core.ml_legacy import *

METHOD_GBC = "gbc"
METHOD_SVM = "svm"
METHOD_RAF = "raf"
METHOD_KNN = "knn"
METHOD_MLP = "mlp"

METHODS = [
    METHOD_GBC,
    METHOD_SVM,
    METHOD_RAF,
    METHOD_KNN,
    METHOD_MLP
]

def parse_arguments():
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-m", type=str, help=f"method from {METHODS}", default=METHOD_GBC)
    args = parser.parse_args()
    return args.m

if __name__ == '__main__':
    method = parse_arguments()

    # -----------------------------------------
    path_train = "./data/train-clean.csv"
    path_test = "./data/test-clean.csv"
    path_output = "./data/submission-01.csv"

    col_types = {
        "HomePlanet": TabularDataset.FEATURE_CATEGORICAL,
        "CryoSleep": TabularDataset.FEATURE_BOOLEAN,
        "Cabin-1": TabularDataset.FEATURE_CATEGORICAL,
        "Cabin-2": TabularDataset.FEATURE_INT,
        "Cabin-3": TabularDataset.FEATURE_CATEGORICAL,
        "Destination": TabularDataset.FEATURE_CATEGORICAL,
        "Age": TabularDataset.FEATURE_INT,
        "VIP": TabularDataset.FEATURE_BOOLEAN,
        "RoomService": TabularDataset.FEATURE_FLOAT,
        "FoodCourt": TabularDataset.FEATURE_FLOAT,
        "ShoppingMall": TabularDataset.FEATURE_FLOAT,
        "Spa": TabularDataset.FEATURE_FLOAT,
        "VRDeck": TabularDataset.FEATURE_FLOAT,
        "Transported": TabularDataset.FEATURE_BOOLEAN,
    }

    col_nan_policy = {
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

    # -----------------------------------------

    dataset_train = TabularDataset()
    dataset_train.load_from_csv(path_train, col_name_label, col_types, col_nan_policy)
    info_train = dataset_train.get_table_info()

    dataset_test = TabularDataset()
    dataset_test.load_from_csv(path_test, col_name_label, col_types)
    info_test = dataset_test.get_table_info()
    for col_name, policy in col_nan_policy.items():
        if policy == TabularDataset.NAN2DATA_DISTRIBUTION:
            distribution = dataset_train.get_feature_distribution(col_name)
            dataset_test.alter_nan2data(col_name, policy, distribution)
        else:
            dataset_test.alter_nan2data(col_name, policy)


    for col_name, policy in col_types.items():
        if policy == TabularDataset.FEATURE_CATEGORICAL:
            output = dataset_train.compute_feature_label_distribution(col_name)
            #plot_feature_label_distribution(output, col_name, f"./data/cat_label_{col_name}.png")


    encoding_table = dataset_train.alter_encode_table()
    dataset_test.alter_encode_table(encoding_table)

    scaler = dataset_train.alter_scale_features()
    dataset_test.alter_scale_features(scaler)


    if method == METHOD_GBC:
        y_pr = mll_train_boosting(dataset_train.features, dataset_train.labels, dataset_test.features)
    elif method == METHOD_SVM:
        y_pr = mll_train_svm(dataset_train.features, dataset_train.labels, dataset_test.features, "rbf")
    elif method == METHOD_RAF:
        y_pr = mll_train_random_forest(dataset_train.features, dataset_train.labels, dataset_test.features)
    elif method == METHOD_KNN:
        y_pr = mll_train_knn(dataset_train.features, dataset_train.labels, dataset_test.features, 10)
    elif method == METHOD_MLP:
        y_pr = mll_train_mlp(dataset_train.features, dataset_train.labels, dataset_test.features, [16, 8])

    f = open(f"./data/output-{method}.csv", "w")
    f.write("PassengerId,Transported\n")
    for i in range(0, y_pr.shape[0]):
        id = dataset_test.df["PassengerId"][i]
        label = "True" if y_pr[i] == 1 else "False"
        f.write(str(id)+","+label+"\n")
    f.close()

