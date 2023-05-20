import json
import os
import pickle
from collections import Counter

from bs4 import BeautifulSoup
from nltk.downloader import download
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from helpers.database import InvertedIndexDatabase
from helpers.inverted_index import InvertedIndex


class Indexer:
    def __init__(self, restart=False) -> None:
        # Create the inverted index, url map, stemmer, and statistics objects.
        self.database = InvertedIndexDatabase("database.db", restart)
        self.database.open()

        self.url_map = dict()
        self.stemmer = PorterStemmer()

        # If the restart flag is set, delete the inverted index file.
        if restart:
            if os.path.exists("url_map.pickle"):
                os.remove("url_map.pickle")

            if os.path.exists("statistics.pickle"):
                os.remove("statistics.pickle")

        if os.path.exists("url_map.pickle"):
            # If the inverted index file exists, load it.
            with open("url_map.pickle", "rb") as f:
                self.url_map = pickle.load(f)

        if not os.path.exists("url_map.pickle") or restart:
            self.create_index()

    def close(self) -> None:
        if self.database is not None:
            self.database.close()

    def create_index(self) -> None:
        # The document id is used to keep track of which document is being
        # processed. It is used as the key for the url map and the value
        # for the inverted index.
        document_id = 0
        inverted_index = InvertedIndex()

        # Recursively loop through all the files in the data directory
        # and add them to the inverted index.
        for subdir, _, files in os.walk("data"):
            for file in files:
                file_path = os.path.join(subdir, file)
                if file_path.endswith(".json"):
                    if document_id % 10000 == 0:
                        self.database.set(inverted_index)
                        inverted_index = InvertedIndex()

                    with open(file_path, "r") as f:
                        data = json.load(f)
                        url = data["url"]
                        self.url_map[document_id] = url

                        soup = BeautifulSoup(data["content"], "html.parser")
                        text = soup.get_text()

                        # Tokenize the text and stem each word.
                        tokens = word_tokenize(text)
                        stemmed_tokens = [self.stemmer.stem(token) for token in tokens]

                        # Count the frequency of each word.
                        word_counts = Counter(stemmed_tokens)

                        # Add the word counts to the inverted index.
                        for word, count in word_counts.items():
                            inverted_index.add(word, document_id, count)

                        document_id += 1

        self.database.set(inverted_index)

        # # If the statistics directory doesn't exist, create it.
        # if not os.path.exists("statistics"):
        #     os.makedirs("statistics")

        # # Create the statistics file.
        # self.statics.create_statistics(self.inverted_index, self.url_map)

        # # Save the inverted index, url map, and statistics objects.
        # with open("inverted_index.pickle", "wb") as f:
        #     pickle.dump(self.inverted_index, f)

        with open("url_map.pickle", "wb") as f:
            pickle.dump(self.url_map, f)

        # with open("statistics.pickle", "wb") as f:
        #     pickle.dump(self.statics, f)

    def merge_list(
        self, list1: list[tuple[int, int]], list2: list[tuple[int, int]]
    ) -> list:
        # Merge two lists of document ids.
        merged_list = []

        i, j = 0, 0

        while i < len(list1) and j < len(list2):
            if list1[i][0] == list2[j][0]:
                merged_list.append(list1[i])
                i += 1
                j += 1
            elif list1[i][0] < list2[j][0]:
                i += 1
            else:
                j += 1

        return merged_list

    def search(self, query: str) -> list:
        if self.database is None:
            raise Exception("Database is not open.")

        # Tokenize the query and stem each word.
        tokens = word_tokenize(query)
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens]

        if not stemmed_tokens:
            return []

        # Get the document ids for each word in the query.
        document_ids: list[tuple[int, int]] = self.database.get(stemmed_tokens[0])
        for token in stemmed_tokens[1:]:
            if len(document_ids) == 0:
                break
            document_ids = self.merge_list(document_ids, self.database.get(token))

        # Get the urls for each document id.
        urls = []
        for posting in document_ids:
            urls.append(self.url_map[int(posting[0])])

        return urls
