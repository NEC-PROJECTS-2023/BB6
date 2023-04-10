from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__, template_folder='template')
model = pickle.load(open("model.pkl", "rb"))


@app.route('/')
def home():
    return render_template('index.html')


@app.route("/predict", methods=["POST"])
def predict():
    user = (request.form.get('text'))

    pre = str((model.predict(user)))

    return pre


if __name__ == "__main__":
    app.run(debug=True)
