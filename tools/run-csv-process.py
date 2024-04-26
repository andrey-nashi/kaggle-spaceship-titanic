


def run(path_csv_in, path_csv_out, header_out):
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

        f_out.write(",".join(feature_vector) + "\n")

    f_in.close()
    f_out.close()



path_in = "../data/train.csv"
path_out = "../data/train-clean.csv"
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
run(path_in, path_out, header_out)