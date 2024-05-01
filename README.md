# kaggle-spaceship-titanic
Spaceship Titanic Kaggle competition

[Kaggle competition](https://www.kaggle.com/competitions/spaceship-titanic)

----------------------------------------------------
1. Place `train.csv`, `test.csv` to data directory
2. Run `python3 run-csv-process.py` to generate new CSV files (`train-clean.csv`, `test-clean.csv`)
   - Split cabin into 3 separate columns
   - Split passenger info into surname and name
   - Complete the test set with label column, the default value set to `False`
3. Run `python3 run-ml-legacy.py -m <method_name>` to generate submission `./data/output-<method_name>.cs`.
Supported methods: `gbc, mlp, knn, svm, raf`

----------------------------------------------------

Exploratory Data Analysis

* Total number of feature vectors in the training set `8693`, test set `4277`
* Column names excluding ID, surname/name: 
'HomePlanet', 'CryoSleep', 'Cabin-1', 'Cabin-2', 'Cabin-3', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported'
* NAN values are handled as distribution for Cabin-1, Cabin-3, Destination, Home planet, Age, for other non-categorical features
the NAN value is set to zero

| Feature name   | Chart (training )                        |
|----------------|------------------------------------------|
| Cabin-1 (deck) | ![img](./data/cat_label_Cabin-1.png)     |
| Cabin-3 (type) | ![img](./data/cat_label_Cabin-3.png)     |
| Destination    | ![img](./data/cat_label_Destination.png) |
| Home planet    | ![img](./data/cat_label_HomePlanet.png)  |

----------------------------------------------------
## Results

| Method            | Result  |
|-------------------|---------|
| K-neighbors       | 0.78442 |
| MLP               | 0.79307 |
| Random forest     | 0.79541 |
| SVM (rbf)         | 0.80196 |
| Gradient boosting | 0.80336 |

