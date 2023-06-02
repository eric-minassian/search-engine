import json
import os
import pickle
from configparser import ConfigParser

from bs4 import BeautifulSoup
from nltk.downloader import download
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from helpers.database import InvertedIndexDatabase
from helpers.inverted_index import InvertedIndex


class Indexer:
    def __init__(self, config_file: ConfigParser, restart=False) -> None:
        self.database_file = config_file.get("DATABASE", "DatabaseFile")
        self.database_index_file = config_file.get("DATABASE", "DatabaseIndexFile")
        self.url_map_file = config_file.get("DATABASE", "UrlMapFile")

        self.bold_weight = config_file.getfloat("INDEXER", "BoldWeight")
        self.title_weight = config_file.getfloat("INDEXER", "TitleWeight")
        self.header_weight = config_file.getfloat("INDEXER", "HeaderWeight")

        self.config_file = config_file

        # Create the inverted index, url map, stemmer, and statistics objects.
        self.database = InvertedIndexDatabase(
            self.database_file, self.database_index_file, restart
        )
        self.database.open()

        self.url_map = dict()
        self.stemmer = PorterStemmer()

        # If the restart flag is set, delete the inverted index file.
        if restart:
            if os.path.exists(self.url_map_file):
                os.remove(self.url_map_file)

        if os.path.exists(self.url_map_file):
            # If the inverted index file exists, load it.
            with open(self.url_map_file, "rb") as f:
                self.url_map = pickle.load(f)

        else:
            # If the inverted index file does not exist, create it.
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
                        # word_counts = Counter(stemmed_tokens)
                        # word_counts_dict = dict(word_counts)
                        word_counts: dict[str, float] = dict()
                        for token in stemmed_tokens:
                            word_counts[token] = word_counts.get(token, 0) + 1

                        # Add additional values for words in title, headers, and bold.
                        bolded_words = ""
                        title_words = ""
                        header_words = ""

                        for bold in soup.find_all("b"):
                            bolded_words += str(bold.get_text()) + " "

                        for title in soup.find_all("title"):
                            title_words += str(title.get_text()) + " "

                        for header in soup.find_all(["h1", "h2", "h3", "h4", "h5"]):
                            header_words += str(header.get_text()) + " "

                        bolded_words = word_tokenize(bolded_words)
                        title_words = word_tokenize(title_words)
                        header_words = word_tokenize(header_words)

                        bolded_words = [
                            self.stemmer.stem(word) for word in bolded_words
                        ]
                        title_words = [self.stemmer.stem(word) for word in title_words]
                        header_words = [
                            self.stemmer.stem(word) for word in header_words
                        ]

                        for word in bolded_words:
                            word_counts[word] = (
                                word_counts.get(word, 0) + self.bold_weight
                            )

                        for word in title_words:
                            word_counts[word] = (
                                word_counts.get(word, 0) + self.title_weight
                            )

                        for word in header_words:
                            word_counts[word] = (
                                word_counts.get(word, 0) + self.header_weight
                            )

                        # Add the word counts to the inverted index.
                        for word, count in word_counts.items():
                            inverted_index.add(word, document_id, count)

                        document_id += 1

        self.database.set(inverted_index)

        self.database.convert_to_tf_idf()

        with open(self.url_map_file, "wb") as f:
            pickle.dump(self.url_map, f)

    def get_postings(self, term: str) -> list:
        return self.database.get(term)

    def get_doc(self, doc_id: int) -> str:
        return self.url_map[doc_id]
