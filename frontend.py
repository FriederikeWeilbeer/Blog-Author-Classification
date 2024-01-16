import joblib
import pandas as pd
from flask import Flask, render_template, url_for, request, redirect

import main

app = Flask(__name__)

@app.route('/')
def index():
	return render_template("index.html", result="", probabilities=dict(), model_option="", status_code=1)


@app.route('/process',methods=["POST", "GET"])
def process():

	if request.method == "POST":

		raw_text = request.form["rawtext"]
		model_option = request.form["model_option"]

		file_path = "models/" + str(model_option.lower()) + ".joblib"
		try:
			model = joblib.load(file_path)
			cleaned_text = main.clean_text(raw_text)
			text = pd.Series(cleaned_text)
			result = model.predict(text)[0]
			probability = model.predict_proba(text)

			labels = model.classes_
			probabilities = dict()

			for i, label in enumerate(labels):
				probabilities[label] = round(probability[0][i] * 100, 2)

			return render_template("index.html", result=result, probabilities=probabilities, model_option=model_option,
								   status_code=0)
		except FileNotFoundError:
			return render_template("index.html", result="", probabilities=dict(), model_option="",
								   status_code=2)

	else:
		redirect("/", code=302)

		return render_template("index.html", result="", probabilities=dict(), model_option="", status_code=1)


if __name__ == '__main__':
	app.run(debug=True, port=5001)
