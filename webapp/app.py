from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")

def index():
    headline = "Hello world"
    return render_template("index.html", headline=headline)


@app.route("/shop")

def shop():
    headline = "Shop here!"
    return render_template("index.html", headline=headline)
