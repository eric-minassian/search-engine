import hashlib
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

        self.simhash_threshold = config_file.getfloat("INDEXER", "SimhashThreshold")

        self.config_file = config_file

        # Create the inverted index, url map, stemmer, and statistics objects.
        self.database = InvertedIndexDatabase(
            self.database_file, self.database_index_file, restart
        )
        self.database.open()

        self.url_map = dict()
        self.stemmer = PorterStemmer()
        self.simhashes = dict()

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

                        # Calculate the simhash for the document.
                        simhash = self.get_simhash(stemmed_tokens)

                        # If the document is not unique, skip it.
                        if self.is_unique(simhash):
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

                            for bold in soup.find_all(["b", "strong"]):
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
                            title_words = [
                                self.stemmer.stem(word) for word in title_words
                            ]
                            header_words = [
                                self.stemmer.stem(word) for word in header_words
                            ]

                            for word in bolded_words:
                                word_counts[word] = word_counts.get(word, 0) + (
                                    self.bold_weight - 1
                                )

                            for word in title_words:
                                word_counts[word] = word_counts.get(word, 0) + (
                                    self.title_weight - 1
                                )

                            for word in header_words:
                                word_counts[word] = word_counts.get(word, 0) + (
                                    self.header_weight - 1
                                )

                            # Add the word counts to the inverted index.
                            for word, count in word_counts.items():
                                inverted_index.add(word, document_id, count)

                        self.simhashes[document_id] = simhash
                        document_id += 1

        self.database.set(inverted_index)

        self.database.convert_to_tf_idf()

        with open(self.url_map_file, "wb") as f:
            pickle.dump(self.url_map, f)

    def get_hash(self, token: str) -> int:
        # Get the hash of the token
        return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)

    def get_simhash(self, tokenize_data: list[str], hashbits=128) -> int:
        # Calculate the simhash of the tokenized data

        # Initialize the vector
        v = [0] * hashbits

        # Loop through each token in the tokenized data
        for t in [self.get_hash(token) for token in tokenize_data]:
            # Loop through each bit in the hash
            for i in range(hashbits):
                bitmask = 1 << i
                if t & bitmask:
                    v[i] += 1
                else:
                    v[i] -= 1

        # Calculate the fingerprint
        fingerprint = 0
        for i in range(hashbits):
            if v[i] >= 0:
                fingerprint += 1 << i

        return fingerprint

    def similarity(self, hash1: int, hash2: int, hashbits=128) -> float:
        # Calculate the similarity between two hashes

        # If the hashes are the same then return 1.0
        if hash1 == hash2:
            return 1.0

        # Calculate the similarity between the hashes
        return 1.0 - (bin(hash1 ^ hash2).count("1") / hashbits)

    def is_unique(self, simhash: int) -> bool:
        for url, hash in self.simhashes.items():
            if self.similarity(simhash, hash) >= self.simhash_threshold:
                return False

        return True

    def get_postings(self, term: str) -> list:
        return self.database.get(term)

    def get_doc(self, doc_id: int) -> str:
        return self.url_map[doc_id]

    def close(self) -> None:
        if self.database is not None:
            self.database.close()
