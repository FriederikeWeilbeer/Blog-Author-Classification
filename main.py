import hashlib
import os


import random

import joblib
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score

from sklearn.preprocessing import MinMaxScaler

import Preprocessing as Preprocessing
from util import Analyse as Analyse
from util.DenseTransformer import DenseTransformer
from util.Logger import Logger
from util.PrintColors import PrintColors
from util.Evaluation import ComparisonAttribute, EvaluationResult, Evaluation

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

LOGGER = Logger(True)


def main():
    """
    Main function
    """

    LOGGER.log("Start")

    # Change the configuration here, you can also generate configurations as a dict
    configuration = {
        "column": "age",
        "max_rows": COMPLETE_DATA_LENGTH,
        "sklearn_steps": [TfidfVectorizer(), LogisticRegression(max_iter=1000, C=0.5, n_jobs=18)],
        "test_split_percentage": 0.3,
        "shuffle": True,
        "shuffle_state": RANDOM_STATE,
        "evaluate": True,
        "evaluate_dist": True,
        "generate_model": True,
        "overwrite": True
    }

    preprocess_config = {
        "max_rows": COMPLETE_DATA_LENGTH,
        "column": "gender",
        "evaluate_dist": True
    }

    configurations = [
        {
            "sklearn_steps": [TfidfVectorizer(), LogisticRegression(max_iter=1000, C=0.5, n_jobs=18)],
        },
        {
            "sklearn_steps": [TfidfVectorizer(max_features=2000), DenseTransformer(), MinMaxScaler(),
                              SVC(max_iter=20000, tol=1e-3, C=0.5, verbose=True, cache_size=5000, kernel="linear")]
        },
        #     {
        #         "max_rows": COMPLETE_DATA_LENGTH / 10,
        #          "column": "gender",
        #          "sklearn_steps": [TfidfVectorizer(), SVC(max_iter=5000, tol=1e-3, C=0.5, verbose=True)]
        #      },
        #     {
        #         "max_rows": COMPLETE_DATA_LENGTH / 10,
        #         "column": "gender",
        #         "sklearn_steps": [TfidfVectorizer(), SVC(max_iter=5000, tol=1e-3, C=0.1, verbose=True)]
        #     }
    ]

    # show_best_model(*find_best_model(preprocess_config, configurations, ComparisonAttribute.PRECISION))

    full_pipeline(**configuration)

    # start pipeline with configuration
    # evaluation, states = training_pipeline(**configuration)

    # log(evaluation, color=PrintColors.CYAN, exec_time=False)
    # log(states, color=PrintColors.CYAN, exec_time=False)

    LOGGER.log("Finished program")


def show_best_model(best_key, results):
    model_string = f'Best Model is: Model {best_key}'

    LOGGER.log(model_string, color=PrintColors.GREEN, exec_time=False)
    LOGGER.log(results[best_key], color=PrintColors.GREEN, exec_time=False)


def find_best_model(preprocess_config, configurations, optimization=ComparisonAttribute.ABSOLUTE):
    """
    Function to find the best model configuration  using score evaluation and comparison

    :param preprocess_config: list of configurations for preprocessing data
    :param configurations: list of configurations for training on data
    :param optimization: Comparison attribute for the evaluation (default: ComparisonAttribute.ABSOLUTE)
    """

    default_config = {
        "sklearn_steps": [TfidfVectorizer(), LogisticRegression(max_iter=1000)],
        "test_split_percentage": 0.3,
        "shuffle": True,
        "shuffle_state": RANDOM_STATE,
    }

    fixed_values = {
        "evaluate": True,
        "generate_model": False,
        "overwrite": True
    }

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
            config[key] = value

        config.update(preprocess_config)
        config["data"] = data

        # call training pipeline with configuration
        # ** is used to convert the dict into multiple params, which the training_pipeline() method requires
        # the output of training_pipeline() is an Evaluation Object and the state
        # a EvaluationResult is generated by passing the output of training_pipeline()
        # and converting it to two parameters of the EvaluationResult constructor with the * operator

        result = EvaluationResult(*training_pipeline(**config))

        # log the Evaluation Result
        LOGGER.log("\n" + str(result) + "\n", color=PrintColors.CYAN, exec_time=False)

        # change the compare attribute for comparing evaluations
        result.evaluation.comp_attr = optimization

        # add to result list
        results[i] = result
        LOGGER.log(f'Finished Model with state: {result.state}', color=PrintColors.BLUE)

    return max(results, key=results.get), results


def full_pipeline(column,
                  max_rows,
                  sklearn_steps,
                  test_split_percentage=0.3,
                  shuffle=False,
                  shuffle_state=random.randint(10000, 20000),
                  evaluate=False,
                  evaluate_dist=False,
                  generate_model=False,
                  overwrite=True
                  ):
    data = preprocess_pipeline(max_rows, column, evaluate_dist)
    return training_pipeline(data, column, max_rows, sklearn_steps, test_split_percentage,
                             shuffle, shuffle_state, evaluate, evaluate_dist, generate_model, overwrite)


def preprocess_pipeline(max_rows, column, evaluate_dist=False):
    # Step 1: Import data
    data = Preprocessing.import_data(FILE_PATH, int(max_rows))
    LOGGER.log("Finished importing")

    # Step 2: Preprocess data
    data = Preprocessing.preprocess_data_multiprocessing(data)
    LOGGER.log("Finished preprocessing")

    # Step Optional: Analyse Distribution
    if evaluate_dist:
        Analyse.analyse_distribution(data, ['age', 'gender', 'sign'])
        LOGGER.log("Finished analysing distribution")

    # Step 3: Prepare data with label
    data = Preprocessing.prepare_data_with_label(data, column)
    LOGGER.log("Preparing Data with label")

    return data


def training_pipeline(data,
                      column,
                      max_rows,
                      sklearn_steps,
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

    :param data:
    :param column: column of dataframe which should be trained
    :param max_rows: max rows for import
    :param sklearn_steps: steps for the sklearn model pipeline
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
    state = state_vars

    # Step 4: Split data
    xtrain, xtest, ytrain, ytest = Preprocessing.split_training_data(data, test_split_percentage, shuffle_state, shuffle)

    # Step 5: Train Model
    model = train_model(sklearn_steps, xtrain, ytrain, state, generate_model, overwrite)

    # Step 6: Evaluate Model
    if evaluate:
        return evaluate_model(model, xtest, ytest), state
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

    model = make_pipeline(*sklearn_steps)

    # Generate String representing the configuration of the model
    pipeline_str = str(state)

    # Calculate the hash value out of model state for the model name
    # used to load and save models with the same state
    hash_value = hashlib.sha256(pipeline_str.encode()).hexdigest()

    # generate filename
    file_name = state["column"] + "-" + str(state["max_rows"]) + "-" + hash_value

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

    LOGGER.log("Finished training the model")
    return model


def evaluate_model(model, x_test, y_test):
    """
    Function for evaluating the model with test data

    :param model: a trained model
    :param x_test: the test values which should be predicted
    :param y_test: the correct labels of the test datat
    :return: An Evaluation Object
    """

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average="macro")
    precision = precision_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    Analyse.show_confusion_matrix(y_test, y_pred)

    return Evaluation(accuracy, recall, precision, f1)


FILE_PATH = "assets/blogtext.csv"
COMPLETE_DATA_LENGTH = 681284
RANDOM_STATE = 41236451
LOGGING = True

DEFAULT_CONFIG = {
    "column": "gender",
    "max_rows": COMPLETE_DATA_LENGTH,
    "sklearn_steps": [TfidfVectorizer(), LogisticRegression(max_iter=1000)],
    "test_split_percentage": 0.3,
    "shuffle": False,
    "shuffle_state": RANDOM_STATE,
    "evaluate": False,
    "evaluate_dist": False,
    "generate_model": False,
    "overwrite": True
}

# driver
if __name__ == "__main__":
    main()
