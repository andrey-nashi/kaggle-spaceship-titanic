from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def mll_train_random_forest(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray = None, random_state: int = 42):
    """
    Train a random forest classifier
    :param x_train: numpy array of feature vectors for training
    :param y_train: numpy array of label vectors for training
    :param x_test: numpy array of feature vectors for testing, or None
    :param random_state: freeze random state
    :return:
    * If x_test is None, then would return model
    * If x_test i numpy array, then would return predictions
    """
    model = RandomForestClassifier(random_state=random_state)
    model.fit(x_train, y_train)
    if x_test is None:
        return model

    y_pr = model.predict(x_test)
    return y_pr

def mll_train_svm(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray = None, kernel: str  = "linear"):
    """
    Train SVM classifier
    :param x_train: numpy array of feature vectors for training
    :param y_train: numpy array of label vectors for training
    :param x_test: numpy array of feature vectors for testing, or None
    :param kernel: SVM kernel like 'linear', 'rbf'
    :return:
    * If x_test is None, then would return model
    * If x_test i numpy array, then would return predictions
    """
    model = SVC(kernel=kernel)
    model.fit(x_train, y_train)
    if x_test is None:
        return model
    y_pr = model.predict(x_test)
    return y_pr

def mll_train_knn(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray = None, neighbors: int = 5):
    """
    Train K-neighbors classifier
    :param x_train: numpy array of feature vectors for training
    :param y_train: numpy array of label vectors for training
    :param x_test: numpy array of feature vectors for testing, or None
    :param neighbors: number of neighbors to use
    :return:
    * If x_test is None, then would return model
    * If x_test i numpy array, then would return predictions
    """
    model = KNeighborsClassifier(n_neighbors=neighbors)
    model.fit(x_train, y_train)
    if x_test is None:
        return model
    y_pr =  model.predict(x_test)
    return y_pr

def mll_train_boosting(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray = None):
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    return

def mll_eval_test(y_gt: np.array, y_pr: np.array) -> tuple:
    """
    Evaluate the results of predictions with a classifier
    :param y_gt: ground truth labels
    :param y_pr: predicted labels
    :return: tuple of (confusion matrix, accuracy, [f_score mean, f_score per class])
    """
    stat_cnf = confusion_matrix(y_gt, y_pr)
    stat_accuracy = accuracy_score(y_gt, y_pr)
    stat_fscore = f1_score(y_gt, y_pr, average="weighted")
    stat_fscore_l = f1_score(y_gt, y_pr, average=None)

    return stat_cnf, stat_accuracy, [stat_fscore, stat_fscore_l]


def plot_pca_3d(features, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(features[:, 0], features[:, 1], features[:, 2], c=labels, cmap="viridis")

    ax.set_xlabel("pca_f_1")
    ax.set_ylabel("pca_f_2")
    ax.set_zlabel("pca_f_3")
    ax.set_title("3D PCA Plot")

    legend = fig.colorbar(scatter, ax=ax, orientation="vertical", shrink=0.5)
    legend.set_label("labels")

    plt.show()

def plot_pca_2d(features, labels):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='viridis', edgecolor='k', alpha=0.7)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('2D PCA Plot')
    plt.colorbar(scatter)
    plt.show()

def save_confusion_matrix(cnf_matrix, path_image):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cnf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=np.arange(1, 9), yticklabels=np.arange(1, 9))
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig(path_image, format="png", dpi=50)