from sklearn.model_selection import train_test_split

from ds_tabular import *
from ml_legacy import *


path_train = "../data/train-clean.csv"
path_test = "../data/test-clean.csv"
path_output = "../data/submission-01.csv"

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


dataset_train = TabularDataset()
dataset_train.load_from_csv(path_train, value_types=header_out)
dataset_test = TabularDataset()
dataset_test.load_from_csv(path_test, value_types=header_out, is_skip_nan=False)


features = dataset_train.get_all_features()
labels = dataset_train.get_all_labels()
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

model = mll_train_random_forest(features, labels)
#stat_cnf, stat_accuracy, f_score = mll_eval_test(y_test, y_pr)
#print(stat_accuracy, f_score)


f = open(path_output, "w")
f.write("PassengerId,Transported\n")
for i in range(len(dataset_test)):
    sample = dataset_test.__getitem__(i)
    sample_origin = dataset_test.get_item_og(i)

    fv = sample[0]
    label = model.predict([fv])[0]
    f.write(str(sample_origin[0]) + "," + str(bool(label)) + "\n")
    print(sample_origin[0], bool(label))

f.close()

