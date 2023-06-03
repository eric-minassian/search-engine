import configparser
import os
import time
from argparse import ArgumentParser

from flask import Flask, render_template, request
from nltk import download

from helpers.indexer import Indexer
from helpers.search import Search


def create_flask_app():
    app = Flask(__name__)
    download("punkt")

    global indexer

    parser = ArgumentParser(description="Search Engine")
    parser.add_argument(
        "--restart",
        action="store_true",
        help="restart the indexer from scratch",
    )
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read("config.ini")

    valid_files = os.path.exists(
        config.get("DATABASE", "DatabaseFile")
    ) and os.path.exists(
        config.get("DATABASE", "DatabaseIndexFile")
        and os.path.exists(config.get("DATABASE", "UrlMapFile"))
    )

    if args.restart or not valid_files:
        print("Restarting indexer...")
        indexer = Indexer(config_file=config, restart=True)
    else:
        print("Loading indexer...")
        indexer = Indexer(config_file=config)

    search = Search(indexer=indexer)

    @app.route("/", methods=["GET", "POST"])
    def home():
        if request.method == "POST":
            search_query = request.form["search_query"]
            start_time = time.time()
            urls = search.search(search_query)
            end_time = time.time()
            return render_template(
                "index.html",
                query=search_query,
                urls=urls,
                time=end_time - start_time,
                count=len(urls),
            )
        return render_template("index.html", query="", urls=[], time=0, count=0)

    return app


if __name__ == "__main__":
    app = create_flask_app()
    try:
        app.run()
    finally:
        indexer.close()
