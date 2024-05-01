def run(path_csv_in, path_csv_out, header_out, extra: tuple = None):
    f_in = open(path_csv_in, "r")
    f_out = open(path_csv_out, "w")

    header_in = (f_in.readline()).strip().split(",")
    target_dim = len(header_out)
    header_out = ",".join(header_out)
    f_out.write(header_out + "\n")

    index_cabin = header_in.index("Cabin")
    index_name = header_in.index("Name")

    for line in f_in:
        shards = line.strip().split(",")
        feature_vector = []
        for index, value in enumerate(shards):
            if index == index_cabin:
                if "/" in value:
                    cabin_info = value.split("/")
                    feature_vector.append(cabin_info[0])
                    feature_vector.append(cabin_info[1])
                    feature_vector.append(cabin_info[2])
                else:
                    feature_vector.append("")
                    feature_vector.append("")
                    feature_vector.append("")
            elif index == index_name:
                if " " in value:
                    name_info = value.split(" ")
                    feature_vector.append(name_info[0])
                    feature_vector.append(name_info[1])
                else:
                    feature_vector.append("")
                    feature_vector.append("")
            else:
                feature_vector.append(value)

        if extra is not None:
            feature_vector += extra

        f_out.write(",".join(feature_vector) + "\n")

    f_in.close()
    f_out.close()

if __name__ == '__main__':
    header_out = [
        "PassengerId",
        "HomePlanet",
        "CryoSleep",
        "Cabin-1",
        "Cabin-2",
        "Cabin-3",
        "Destination",
        "Age",
        "VIP",
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck",
        "Name",
        "Surname",
        "Transported"
    ]

    path_train_in = "../data/train.csv"
    path_train_out = "../data/train-clean.csv"
    run(path_train_in, path_train_out)

    path_test_in = "../data/test.csv"
    path_test_out = "../data/test-clean.csv"
    run(path_test_in, path_test_out, header_out, extra=["False"])
