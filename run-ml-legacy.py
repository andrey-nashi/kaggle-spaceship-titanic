from sklearn.model_selection import train_test_split
from tqdm import tqdm

from core.ds_tabular import *
from core.ml_legacy import *
from core.ml_plotter import *

if __name__ == '__main__':

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
            plot_feature_label_distribution(output, col_name, f"./data/cat_label_{col_name}.png")
            print("---------------------")
            print(col_name)
            print(output)





    #print(info_test)



