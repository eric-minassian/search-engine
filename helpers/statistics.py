import os

from helpers.inverted_index import InvertedIndex


class Statistics:
    def __init__(self) -> None:
        self._num_documents = 0
        self._num_unique_words = 0
        self._index_size = 0

        # If the statistics directory doesn't exist, create it.
        if not os.path.exists("statistics"):
            os.makedirs("statistics")

    def create_statistics(self, inverted_index: InvertedIndex, url_map: dict) -> None:
        self._num_documents = len(url_map)
        self._num_unique_words = len(inverted_index)
        self._index_size = inverted_index.size()

        with open("statistics/statistics.txt", "w") as f:
            f.write(str(self))

    def add_document(self) -> None:
        self._num_documents += 1

    def add_unique_word(self) -> None:
        self._num_unique_words += 1

    def add_index_size(self, size: int) -> None:
        self._index_size += size

    def set_document_count(self, count: int) -> None:
        self._num_documents = count

    def set_unique_word_count(self, count: int) -> None:
        self._num_unique_words = count

    def set_index_size(self, size: int) -> None:
        self._index_size = size

    def __str__(self) -> str:
        # Return a table of the statistics.
        return (
            f"{'Number of documents':<30} {self._num_documents:>10}\n"
            f"{'Number of unique words':<30} {self._num_unique_words:>10}\n"
            f"{'Index size (in bytes)':<30} {self._index_size:>10}"
        )
