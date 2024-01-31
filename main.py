import hashlib
import os
import random
import traceback

import joblib
import pandas as pd

import Preprocessing as Preprocessing
from util import Analyse as Analyse
from util.Logger import Logger
from util.PrintColors import PrintColors
from util.Evaluation import ComparisonAttribute, EvaluationResult, Evaluation

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
LOGGER = Logger(True)


def main():
    """
    Main function
    """

    LOGGER.log("Start")

    # Configuration for exporting the Model
    # Copy the best model output from the console into below and set generate_model to True
    configuration = \
        {
            "model_name": "LR-C05-SAGA-tfidf_base",
            "column": "gender",
            "max_rows": 681284,
            "sklearn_steps": [TfidfVectorizer(min_df=5, max_df=0.9, sublinear_tf=True),
                              LogisticRegression(C=0.5, max_iter=1000, n_jobs=14, solver='saga')],
            "test_split_percentage": 0.2,
            "shuffle": True,
            "shuffle_state": 41236451,
            "evaluate": True,
            "evaluate_dist": True,
            "generate_model": True,
            "overwrite": False
        }

    # export model
    full_pipeline(**configuration)

    # preprocess config for multiple model training -> So the preprocessing isnt done each turn
    preprocess_config = {
        "max_rows": COMPLETE_DATA_LENGTH,
        "normalize_data": False,
        "column": "gender",
        "evaluate_dist": True
    }

    # list of configurations for training and testing against each other
    configurations = [
        {
            "model_name": "LR-C05-SAGA-df_5-09_sublin-max_no",
            "sklearn_steps": [TfidfVectorizer(max_df=0.9, min_df=5, sublinear_tf=True),
                              LogisticRegression(C=0.5, max_iter=1000, n_jobs=18, solver='saga')],
        },
        {
            "model_name": "LR-C05-SAGA-df_5-09_sublin-max_no",
            "sklearn_steps": [TfidfVectorizer(max_df=0.9, min_df=5, sublinear_tf=True),
                              LogisticRegression(C=0.5, max_iter=1000, n_jobs=18, solver='saga')],
        },
    ]

    # start find best model and show the results
    show_best_model(*find_best_model(preprocess_config, configurations, ComparisonAttribute.PRECISION, export=True))

    LOGGER.log("Finished program")


def show_best_model(best_key, results):
    """
    Function for showing the best model in the console

    :param best_key: the key of the best model in the result dict
    :param results: a dict with all the results as EvalauationResults
    """
    best = results[best_key]

    model_string = f'Best Model is: Model [{best_key}] {best.state["model_name"]}'

    LOGGER.log(model_string, color=PrintColors.GREEN, exec_time=False)
    LOGGER.log(results[best_key], color=PrintColors.GREEN, exec_time=False)
    LOGGER.log(f'Best model export: \n{best.export_config()}', color=PrintColors.BLUE, exec_time=False, current_time=False)


def find_best_model(preprocess_config, configurations, optimization=ComparisonAttribute.ABSOLUTE, export=False):
    """
    Function to find the best model configuration  using score evaluation and comparison

    :param preprocess_config: list of configurations for preprocessing data
    :param configurations: list of configurations for training on data
    :param optimization: Comparison attribute for the evaluation (default: ComparisonAttribute.ABSOLUTE)
    :param export: If the evaluation should be exported to a csv file (default: False)
    """

    # default config if config item was not given
    default_config = {
        "sklearn_steps": [TfidfVectorizer(), LogisticRegression(max_iter=1000)],
        "test_split_percentage": 0.2,
        "shuffle": True,
        "shuffle_state": RANDOM_STATE,
        "model_name": "NO_NAME"
    }

    # fixed config values for overwriting
    fixed_values = {
        "evaluate": True,
        "generate_model": False,
        "overwrite": True
    }

    # preprocess the data
    data = preprocess_pipeline(**preprocess_config)

    results = dict()

    # iterate over all configurations
    for i, config in enumerate(configurations):

        # check if all config values are given, if not add them from the default configuration
        for key, value in default_config.items():
            if key not in config.keys():
                config[key] = value

        # add all fixed configuration values to the config
        for key, value in fixed_values.items():

            if key == "model_name":
                config[key] = value + i
            else:
                config[key] = value

        # add preprocessing config to complete config
        config.update(preprocess_config)

        # add data to config
        config["data"] = data

        try:
            # call training pipeline with configuration
            # ** is used to convert the dict into multiple params, which the training_pipeline() method requires
            # the output of training_pipeline() is an Evaluation Object and the state
            # a EvaluationResult is generated by passing the output of training_pipeline()
            # and converting it to two parameters of the EvaluationResult constructor with the * operator
            result = EvaluationResult(*training_pipeline(**config))

            # change the compare attribute for comparing evaluations
            result.evaluation.comp_attr = optimization

            # add to result list
            results[i] = result

            if export:
                # export evaluation to csv
                Analyse.output_evaluation(result, preprocess_config["column"])

            LOGGER.log(f'Finished Model with state: {result.state}', color=PrintColors.BLUE)
        except Exception:
            tb = traceback.format_exc()
            LOGGER.log("Model Failed with Error: " + tb, color=PrintColors.RED)

    return max(results, key=results.get), results


def full_pipeline(model_name,
                  column,
                  max_rows,
                  sklearn_steps,
                  normalize_data=False,
                  test_split_percentage=0.3,
                  shuffle=False,
                  shuffle_state=random.randint(10000, 20000),
                  evaluate=False,
                  evaluate_dist=False,
                  generate_model=False,
                  overwrite=True
                  ):
    """
    Complete Pipeline for exporting a model

    :param model_name: Name of the model
    :param column: column of dataframe which should be trained
    :param max_rows: max rows for import
    :param sklearn_steps: steps for the sklearn model pipeline
    :param normalize_data: whether to normalize the data in preprocessing
    :param test_split_percentage: percentage of test data
    :param shuffle: flag for shuffling data
    :param shuffle_state: state to reproducing the shuffle process
    :param evaluate: flag if the model should be evaluated (default: False)
    :param evaluate_dist: flag if a distribution should be generated (default: False)
    :param generate_model: flag if a model output should be generated (default: False)
    :param overwrite: flag for overwriting existing data (default: True)
    :return: An Evaluation object for the training pipeline and the state or None if evaluate is False
    """
    data = preprocess_pipeline(max_rows, column, normalize_data, evaluate_dist)
    return training_pipeline(model_name, data, column, max_rows, sklearn_steps, normalize_data, test_split_percentage,
                             shuffle, shuffle_state, evaluate, evaluate_dist, generate_model, overwrite)


def preprocess_pipeline(max_rows, column, normalize_data, evaluate_dist=False):
    """
    Complete Pipeline for exporting a model

    :param max_rows: max rows for import
    :param column: column of dataframe which should be trained
    :param normalize_data: whether to normalize the data in preprocessing
    :param evaluate_dist: flag if a distribution should be generated (default: False)
    :return: preprocessed dataframe
    """

    # Step 1: Import data
    data = Preprocessing.import_data(FILE_PATH, int(max_rows))
    LOGGER.log("Finished importing")

    # Step 2: Preprocess data
    data = Preprocessing.preprocess_data_multiprocessing(data)
    LOGGER.log("Finished preprocessing")

    if normalize_data:
        data = Preprocessing.normalize_data(data, column, min(data[column].value_counts()))

    # Step Optional: Analyse Distribution
    if evaluate_dist:
        Analyse.analyse_distribution('total', data, ['age', 'gender', 'sign'], [])
        LOGGER.log("Finished analysing distribution")

    # Step 3: Prepare data with label
    data = Preprocessing.prepare_data_with_label(data, column)
    LOGGER.log("Preparing Data with label")

    return data


def training_pipeline(model_name,
                      data,
                      column,
                      max_rows,
                      sklearn_steps,
                      normalize_data=False,
                      test_split_percentage=0.3,
                      shuffle=False,
                      shuffle_state=random.randint(10000, 20000),
                      evaluate=False,
                      evaluate_dist=False,
                      generate_model=False,
                      overwrite=True
                      ):
    """
    Function for the training pipeline

    :param model_name: name of the model
    :param data: preprocessed data as a dataframe
    :param column: column of dataframe which should be trained
    :param max_rows: max rows for import
    :param sklearn_steps: steps for the sklearn model pipeline
    :param normalize_data: whether to normalize the data in preprocessing
    :param test_split_percentage: percentage of test data
    :param shuffle: flag for shuffling data
    :param shuffle_state: state to reproducing the shuffle process
    :param evaluate: flag if the model should be evaluated (default: False)
    :param evaluate_dist: flag if a distribution should be generated (default: False)
    :param generate_model: flag if a model output should be generated (default: False)
    :param overwrite: flag for overwriting existing data (default: True)
    :return: An Evaluation object for the training pipeline and the state or None if evaluate is False
    """

    # get the state of the method parameters, used for saving the trained model later
    state_vars = locals()

    state_vars.pop("data")
    state_vars.pop("overwrite")
    state_vars.pop("evaluate")
    state_vars.pop("evaluate_dist")
    state_vars.pop("generate_model")
    state = state_vars.copy()

    # Step 4: Split data
    xtrain, xtest, ytrain, ytest = Preprocessing.split_training_data(data, test_split_percentage, shuffle_state, shuffle)

    # Step 5: Train Model
    model = train_model(sklearn_steps, xtrain, ytrain, state, generate_model, overwrite)
    training_time = LOGGER.get_duration()

    state["training_time"] = training_time
    LOGGER.log("Finished training the model")

    # Step 6: Evaluate Model
    if evaluate:
        result = evaluate_model(model, xtest, ytest, state_vars["model_name"], column)
        # log the Evaluation Result
        LOGGER.log("\n" + str(result) + "\n", color=PrintColors.CYAN, exec_time=False)

        test_data = pd.DataFrame({column: ytest})

        # show distribution and what percentage was predicted correctly
        Analyse.analyse_distribution(state_vars["model_name"], test_data, [column], result.get_correct_by_category())

        return result, state
    else:
        return None


def train_model(sklearn_steps, x_train, y_train, state, generate_model, overwrite):
    """
    Function for training the model with LogisticRegression

    :param sklearn_steps: steps for the sklearn model pipeline
    :param x_train: a pandas dataframe with all training data
    :param y_train: a pandas dataframe with all labels
    :param state: a string containing the state of the model
    :param generate_model: a flag for saving the model to a file
    :param overwrite: a flag to overwrite existing models
    :return: xtrain, xtest, ytrain, ytest
    """

    LOGGER.log(f'Start training model {state["model_name"]}')

    model = make_pipeline(*sklearn_steps)

    model_name = state.pop("model_name")

    # Generate String representing the configuration of the model
    config_str = str(state)

    # Calculate the hash value out of model state for the model name
    # used to load and save models with the same state
    hash_value = hashlib.sha256(config_str.encode()).hexdigest()

    # generate prefix out of model name
    prefix = model_name + "-" if "NO_NAME" not in model_name else ""

    # generate filename
    file_name = prefix + state["column"] + "-" + str(state["max_rows"]) + "-" + hash_value

    # file path of the saved model
    file_path = "serialized/" + file_name + ".joblib"

    # if the model exists and overwrite is disabled
    if os.path.exists(file_path) and not overwrite:
        model = joblib.load(file_path)

    # if the model does not exist, or overwrite is enabled
    else:
        model.fit(x_train, y_train)
        joblib.dump(model, file_path)

    # export finished model to a file
    if generate_model:
        category = state["column"]
        model_path = "models/" + category + ".joblib"
        joblib.dump(model, model_path)

    state["hash_value"] = hash_value
    state["model_name"] = model_name

    try:
        vocabulary_size = len(model.named_steps['tfidfvectorizer'].vocabulary_)
        state["number_features"] = vocabulary_size
    except Exception as e:
        LOGGER.log(f'Could not find the vocabulary: {str(e)}', color=PrintColors.RED)
    return model


def evaluate_model(model, x_test, y_test, model_name, column):
    """
    Function for evaluating the model with test data

    :param model_name: name of the model
    :param model: a trained model
    :param x_test: the test values which should be predicted
    :param y_test: the correct labels of the test data
    :param column: column of dataframe which should be trained

    :return: An Evaluation Object
    """

    y_pred = model.predict(x_test)

    # show confusion matrix for the test
    Analyse.show_confusion_matrix(y_test, y_pred, model_name, column)

    return Evaluation(x_test, y_test, y_pred)


FILE_PATH = "assets/blogtext.csv"
COMPLETE_DATA_LENGTH = 681284
RANDOM_STATE = 41236451
LOGGING = True

# driver
if __name__ == "__main__":
    main()
