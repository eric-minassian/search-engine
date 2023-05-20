import time
from argparse import ArgumentParser

from flask import Flask, render_template, request
from nltk import download

from helpers.indexer import Indexer

download("punkt")

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        search_query = request.form["search_query"]
        start_time = time.time()
        urls = indexer.search(search_query)
        end_time = time.time()
        return render_template(
            "index.html", query=search_query, urls=urls, time=end_time - start_time
        )
    return render_template("index.html", query="", urls=[], time=0)


if __name__ == "__main__":
    indexer = Indexer()
    try:
        app.run(debug=True)
    finally:
        indexer.close()
