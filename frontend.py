import os.path
import random

import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, jsonify

import Preprocessing

app = Flask(__name__)


FILE_PATH = "assets/blogtext.csv"

def get_text_samples():

	prefix = "models"
	try:
		gender_model = joblib.load(os.path.join(prefix, 'gender.joblib'))
		age_model = joblib.load(os.path.join(prefix, 'age.joblib'))
		sign_model = joblib.load(os.path.join(prefix, 'sign.joblib'))

		data = Preprocessing.import_data(FILE_PATH, 2000)
		preprocessed = Preprocessing.preprocess_data(data.copy())
		data["preprocessed"] = preprocessed["text"]
		data["age"] = preprocessed["age"]

		x_data = data["preprocessed"]

		data["gender_prediction"] = gender_model.predict(x_data)
		data["age_prediction"] = age_model.predict(x_data)
		data["sign_prediction"] = sign_model.predict(x_data)

		# Add columns with 1 if prediction is correct, 0 otherwise
		data["gender_correct"] = (data["gender_prediction"] == data["gender"]).astype(int)
		data["age_correct"] = (data["age_prediction"] == data["age"]).astype(int)
		data["sign_correct"] = (data["sign_prediction"] == data["sign"]).astype(int)

		correct_columns = ["gender_correct", "age_correct", "sign_correct"]

		# Create a mask for rows where exactly one correctness column is equal to 1
		one_correct_mask = data[correct_columns].eq(1).sum(axis=1) == 1
		two_correct_mask = data[correct_columns].eq(1).sum(axis=1) == 2
		all_correct_mask = data[correct_columns].eq(1).sum(axis=1) == 3

		# Filter the DataFrame using the mask
		one_correct = data[one_correct_mask]
		two_correct = data[two_correct_mask]
		all_correct = data[all_correct_mask]

		return all_correct, two_correct, one_correct

	except FileNotFoundError:
		return None


IMPOSSIBLE, HARD, MEDIUM = get_text_samples()


@app.route('/')
def index():
	return render_template("index.html", raw_text="", result="", probabilities=dict(), model_option="", status_code=1)


def get_prediction(model, text):
	result = model.predict(text)[0]
	probability = model.predict_proba(text)

	labels = model.classes_
	probabilities = dict()

	for i, label in enumerate(labels):
		probabilities[label] = round(probability[0][i] * 100, 2)

	sorted_probabilities = dict(sorted(probabilities.items(), key=lambda x: x[1], reverse=True))

	return result, sorted_probabilities



@app.route('/get_new_quote')
def get_new_quote():
	# Logic to get a new quote goes here

	global HARD

	value_list = HARD["text"].tolist()

	# Get a random index within the range of the list length
	random_index = random.randrange(len(value_list))

	new_quote = value_list[random_index]

	return jsonify({'new_quote': new_quote, 'random_index': random_index})


def get_data_point(df, i, column):
	value_list = df[column].tolist()
	return value_list[i]

@app.route('/game', methods=["POST", "GET"])
def game():
	global HARD
	if request.method == "GET":
		return render_template("game.html", status_code=0, result=dict())
	elif request.method == "POST":

		gender = str(request.form["gender"]).lower()
		age = str(request.form["age"]).lower()
		sign = str(request.form["sign"]).lower()

		random_index = request.form["random_index"]

		gender_correct = str(get_data_point(HARD, int(random_index), "gender")).lower()
		gender_predict = str(get_data_point(HARD, int(random_index), "gender_prediction")).lower()
		age_correct = str(get_data_point(HARD, int(random_index), "age")).lower()
		age_predict = str(get_data_point(HARD, int(random_index), "age_prediction")).lower()
		sign_correct = str(get_data_point(HARD, int(random_index), "sign")).lower()
		sign_predict = str(get_data_point(HARD, int(random_index), "sign_prediction")).lower()

		results = {
			"gender_player": gender,
			"gender_cpu": gender_predict,
			"gender_real": gender_correct,
			"age_player": age,
			"age_cpu": age_predict,
			"age_real": age_correct,
			"sign_player": sign,
			"sign_cpu": sign_predict,
			"sign_real": sign_correct,

		}

		return render_template("game.html", status_code=1, result=results)


@app.route('/multi', methods=["POST", "GET"])
def multi():
	raw_text = ""

	if request.method == "GET":
		return render_template("multi.html", raw_text="", result=dict(), status_code=0)

	if request.method == "POST":
		raw_text = request.form["rawtext"]
		cleaned_text = Preprocessing.clean_text(raw_text)
		text = pd.Series(cleaned_text)

		prefix = "models"

		try:
			gender_model = joblib.load(os.path.join(prefix, 'gender.joblib'))
			age_model = joblib.load(os.path.join(prefix, 'age.joblib'))
			sign_model = joblib.load(os.path.join(prefix, 'sign.joblib'))

			result = {
				"gender": get_prediction(gender_model, text),
				"age": get_prediction(age_model, text),
				"sign": get_prediction(sign_model, text),
			}

			return render_template("multi.html", raw_text=raw_text, result=result,
								   status_code=1)

		except FileNotFoundError:
			return render_template("multi.html", raw_text="", result=dict(),
								   status_code=2)


@app.route('/single', methods=["POST", "GET"])
def single():
	raw_text = ""

	if request.method == "POST":

		raw_text = request.form["rawtext"]
		model_option = request.form["model_option"]

		cleaned_text = Preprocessing.clean_text(raw_text)
		text = pd.Series(cleaned_text)

		file_path = "models/" + str(model_option.lower()) + ".joblib"
		try:
			model = joblib.load(file_path)

			result, probabilities = get_prediction(model, text)

			return render_template("index.html", raw_text=raw_text, result=result, probabilities=probabilities,
								   model_option=model_option,
								   status_code=0)
		except FileNotFoundError:
			return render_template("index.html", raw_text="", result="", probabilities=dict(), model_option="",
								   status_code=2)

	else:
		redirect("/", code=302)

		return render_template("index.html", result="", probabilities=dict(), model_option="", status_code=1)


if __name__ == '__main__':
	app.run(debug=True, port=5001)
