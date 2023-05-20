import os
import pickle

from helpers.inverted_index import InvertedIndex


class InvertedIndexDatabase:
    def __init__(self, filename, restart=False):
        self.filename = filename
        self.index = {}
        self.database = None
        self.restart = restart

    def open(self):
        if self.restart:
            if os.path.exists(self.filename):
                os.remove(self.filename)
            if os.path.exists("database_index.pickle"):
                os.remove("database_index.pickle")
            self.database = open(self.filename, "w+")
        else:
            self.database = open(self.filename, "a+")
            if os.path.exists("database_index.pickle"):
                with open("database_index.pickle", "rb") as f:
                    self.index = pickle.load(f)
            else:
                self.refresh_index()

    def close(self):
        if self.database is None:
            raise Exception("Database is not open.")

        self.database.close()

        with open("database_index.pickle", "wb") as f:
            pickle.dump(self.index, f)

    def refresh_index(self):
        if self.database is None:
            raise Exception("Database is not open.")

        self.index = {}

        self.database.seek(0)

        while True:
            seek_pos = self.database.tell()
            line = self.database.readline()

            if not line:
                break

            key = line.split(":")[0]
            self.index[key] = seek_pos

    def set(self, inverted_index: InvertedIndex):
        if self.database is None:
            raise Exception("Database is not open.")

        if inverted_index is None or len(inverted_index) == 0:
            return

        with open("database_temp.db", "w+") as f:
            for key, value in inverted_index.items_str():
                for i in range(len(value)):
                    value[i] = str(value[i])

                if key in self.index:
                    self.database.seek(self.index[key])
                    old_value = self.database.readline().split(":")[1].strip()
                    old_value_list = old_value.split("|")
                    new_value_list = old_value_list + value
                    new_value = "|".join(new_value_list)
                else:
                    new_value = "|".join(value)

                f.write(f"{key}:{new_value}\n")

        self.database.close()
        os.remove(self.filename)
        os.rename("database_temp.db", self.filename)

        self.database = open(self.filename, "a+")
        self.refresh_index()

    def get(self, key) -> list[tuple[int, int]]:
        if self.database is None:
            raise Exception("Database is not open.")

        if key not in self.index:
            return []

        self.database.seek(self.index[key])
        data = self.database.readline().split(":")[1].strip()

        data_list = data.split("|")
        data_tuple_list = list()
        for data in data_list:
            data = data.replace("(", "").replace(")", "")
            temp_tuple = tuple(data.split(", "))
            new_tuple = list()
            for i in range(len(temp_tuple)):
                new_tuple.append(int(temp_tuple[i]))

            data_tuple_list.append(tuple(new_tuple))

        return data_tuple_list


if __name__ == "__main__":
    # Usage example:
    db = InvertedIndexDatabase("database.db")

    db.open()

    inverted_index = InvertedIndex()

    inverted_index.add("hello", 1, 1)
    inverted_index.add("hello", 2, 1)
    inverted_index.add("world", 1, 1)
    inverted_index.add("world", 2, 1)

    db.set(inverted_index)

    print(db.get("hello"))

    inverted_index2 = InvertedIndex()

    inverted_index2.add("hello", 3, 1)
    inverted_index2.add("hello", 4, 1)
    inverted_index2.add("world", 3, 1)
    inverted_index2.add("world", 4, 1)

    db.set(inverted_index2)

    print(db.get("hello"))

    db.close()
