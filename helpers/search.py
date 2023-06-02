from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from helpers.indexer import Indexer


class Search:
    def __init__(self, indexer: Indexer) -> None:
        self.indexer = indexer
        self.stemmer = PorterStemmer()

    def search(self, query: str) -> list:
        # Tokenize the query and stem each word.
        tokens = word_tokenize(query)
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens]

        if not stemmed_tokens:
            return []

        document_ids = set()
        for token in stemmed_tokens:
            document_ids = document_ids.union(self.indexer.get_postings(token))

        document_ids = list(document_ids)

        # Sort the document ids by their tf-idf score.
        document_ids.sort(key=lambda x: x[1], reverse=True)

        # Get the urls for each document id.
        urls = []
        for posting in document_ids:
            urls.append(self.indexer.get_doc(int(posting[0])))

        return urls
