import sys
from typing import ItemsView

from nltk.corpus import words


class InvertedIndex:
    def __init__(self):
        self.index = dict()

    def add(self, term: str, document_id: int, term_frequency: int) -> None:
        if term not in self.index:
            self.index[term] = list()

        self.index[term].append((document_id, term_frequency))

    def get(self, term: str) -> list:
        return self.index.get(term, list())

    def items(self) -> ItemsView[str, list[tuple[int, int]]]:
        return self.index.items()

    def items_str(self) -> ItemsView[str, list[str]]:
        return self.index.items()

    def size(self) -> int:
        return sys.getsizeof(self.index)

    def __len__(self) -> int:
        return len(self.index)

    def __str__(self) -> str:
        return str(self.index)
