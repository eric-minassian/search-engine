import sys
from typing import ItemsView

from nltk.corpus import words


class InvertedIndex:
    def __init__(self):
        self.index = dict()

    def add(self, term: str, document_id: int, term_frequency: float) -> None:
        if term not in self.index:
            self.index[term] = list()

        self.index[term].append((document_id, term_frequency))

    def get(self, term: str) -> list:
        return self.index.get(term, list())

    def items(self) -> ItemsView[str, list[tuple[int, float]]]:
        return self.index.items()

    def size(self) -> int:
        return sys.getsizeof(self.index)

    def __len__(self) -> int:
        return len(self.index)

    def __str__(self) -> str:
        return str(self.index)


if __name__ == "__main__":
    temp = InvertedIndex()
    temp.add("hello", 1, 1.12)
    temp.add("hello", 2, 1)

    temp.add("world", 1, 1)

    items = temp.items()
    for key, value in items:
        print(key, value)
