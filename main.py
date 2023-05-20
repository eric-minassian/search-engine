from argparse import ArgumentParser

from helpers.indexer import Indexer


def main() -> None:
    parser = ArgumentParser(description="Indexer")
    parser.add_argument(
        "--restart",
        action="store_true",
        help="restart the indexer from scratch",
    )
    args = parser.parse_args()

    if args.restart:
        indexer = Indexer(restart=True)
    else:
        indexer = Indexer()

    search_query = input("Enter a search query: ")
    while search_query != "exit":
        urls = indexer.search(search_query)

        print(f"Found {len(urls)} results.")
        for url in urls:
            print(url)

        search_query = input("Enter a search query: ")

    indexer.close()


if __name__ == "__main__":
    main()
