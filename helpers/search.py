from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from helpers.indexer import Indexer


class Search:
    def __init__(self, indexer: Indexer) -> None:
        self.indexer = indexer
        self.stemmer = PorterStemmer()

    def search(self, query: str) -> list[str]:
        # Tokenize the query and stem each word.
        tokens = word_tokenize(query)
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens]

        if not stemmed_tokens:
            return []

        document_ids = dict()
        for token in stemmed_tokens:
            # Get the postings for the token.
            postings = self.indexer.get_postings(token)

            # If there are no postings for the token, skip it.
            if not postings:
                continue

            # Add the postings to the document ids.
            for posting in postings:
                document_ids[posting[0]] = document_ids.get(posting[0], 0) + posting[1]
            
        document_ids = list(document_ids.items())

        # Sort the document ids by their tf-idf score.
        document_ids.sort(key=lambda x: x[1], reverse=True)

        # Get the urls for each document id.
        urls = []
        for posting in document_ids:
            urls.append(self.indexer.get_doc(int(posting[0])))

        return urls
