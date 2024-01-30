import os

import numpy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from nltk import FreqDist
from sklearn.metrics import confusion_matrix
import seaborn as sns


def analyse_distribution(prefix, data_frame, columns, overlapping):
    """
    Function analysing distribution of values in a column of a dataframe

    :param data_frame: dataframe to analyse
    :param columns: an array of columns to analyse
    """
    for column in columns:
        if column in data_frame.columns:
            build_distribution_graph(prefix, data_frame, column, overlapping)
        else:
            raise ValueError("Column '{}' is not in the dataframe".format(column))


def build_distribution_graph(prefix, data_frame, column, overlapping):
    """
    Function for building a distribution graph

    :param data_frame: dataframe to analyse
    :param column: a column to analyse
    """

    # Get the frequency distribution of every value
    freq = FreqDist(np.array([x for x in data_frame[column]]).ravel())

    # set the figure size
    plt.figure(figsize=(13, 7))

    # Sort the frequency data alphabetically by label
    sorted_freq = dict(sorted(freq.items()))

    # Plot bar chart
    bars = plt.bar(sorted_freq.keys(), sorted_freq.values(), edgecolor='black')  # Adjust bins as needed

    if len(overlapping) == len(sorted_freq):
        overlapping_freq = [overlapping[key] for key in sorted_freq.keys()]
        plt.bar(sorted_freq.keys(), overlapping_freq,
                color='green', alpha=0.5, edgecolor='black', label='Percentage')  # Overlapping bars

    # Add absolute values above the bars
    for bar in bars:
        yval = bar.get_height()

        # Values placing the text
        rotation = 'horizontal'
        text = None

        # Calculate the percentage of one bar
        percentage = (yval / sum(freq.values()) * 100)

        # if the chart contains less than 15 x-values, the text is displayed horizontal
        if len(freq.keys()) < 15:
            rotation = 'horizontal'

            # put text in two lines
            text = f"{round(yval, 2)}\n({round(percentage, 2)}%)"

            # add a margin for text above the bars by extending the y-axis by 1/7
            plt.ylim(top=max(freq.values()) + max(freq.values()) / 7)

        # else it is displayed vertical
        else:
            rotation = 'vertical'

            # put text in one line
            text = f"{round(yval, 2)} ({round(percentage, 2)}%)"

            # add a margin for text above the bars by extending the y-axis by 1/3
            plt.ylim(top=max(freq.values()) + max(freq.values()) / 3)

        # display text
        plt.text(bar.get_x() + bar.get_width() / 2, yval + max(freq.values()) / 50, text, ha='center', va='bottom',
                 rotation=rotation)

    # # Add labels and title to diagram
    plt.xlabel(column)
    plt.ylabel('frequency')
    plt.title(column + ' distribution')

    filepath = os.path.join("graphs", prefix, column + "_distribution.png")

    # Extract the directory from the file path
    directory = os.path.dirname(filepath)

    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(directory):
        os.makedirs(directory)

    # save the plot to a file
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()


def show_confusion_matrix(y_test, y_pred, model_name, column):
    # get labels of the data for plot
    unique_labels = sorted(set(y_test))

    # set plot size
    plt.figure(figsize=(16, 9))

    # get confusion matrix of the model
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)

    # generate a heatmap out of the confusion matrix
    heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels,
                          yticklabels=unique_labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    filepath = os.path.join('graphs', model_name, column + 'confusion_matrix_heatmap.png')

    # Extract the directory from the file path
    directory = os.path.dirname(filepath)

    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(directory):
        os.makedirs(directory)

    # save the heatmap to a file
    heatmap.get_figure().savefig(filepath, bbox_inches='tight')
    plt.close()


def output_evaluation(evaluation_result, filename):

    eval_dict = evaluation_result.state
    eval_dict.update(evaluation_result.evaluation.get_metrics())

    # Convert lists to strings
    str_dict = {key: str(val) if isinstance(val, list) or isinstance(val, numpy.ndarray) else val for key, val in eval_dict.items()}

    df = pd.DataFrame(str_dict)

    print(type(str_dict["tn"]))

    # put model name to the beginning
    df.insert(0, "model_name", df.pop("model_name"))

    filename = filename + ".csv"

    # Check if the file exists
    if os.path.exists(filename):
        # Append DataFrame to the existing CSV file
        df.to_csv(filename, sep=",", mode='a', header=False, index=False)
    else:
        # Create a new CSV file
        df.to_csv(filename, sep=",", mode='w', header=True, index=False)
