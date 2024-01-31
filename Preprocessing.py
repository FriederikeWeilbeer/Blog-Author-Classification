import pandas as pd
import multiprocessing as mp
import re

from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

STOP_WORDS = set(stopwords.words('english'))


def normalize_data(dataframe, target_column, desired_count):
    # Calculate the counts of each value in 'target_column'
    value_counts = dataframe[target_column].value_counts()

    # Get the values that need oversampling or undersampling
    values_to_oversample = value_counts[value_counts < desired_count].index.tolist()
    values_to_undersample = value_counts[value_counts >= desired_count].index.tolist()

    # Oversample minority classes
    oversampled_minority = pd.concat([
        resample(dataframe[dataframe[target_column] == value],
                 replace=True,
                 n_samples=desired_count,
                 random_state=42)
        for value in values_to_oversample
    ]) if values_to_oversample else []  # Check if there are values to oversample

    # Undersample majority classes
    undersampled_majority = pd.concat([
        resample(dataframe[dataframe[target_column] == value],
                 replace=False,
                 n_samples=desired_count,
                 random_state=42)
        for value in values_to_undersample
    ]) if values_to_undersample else []  # Check if there are values to undersample

    if len(oversampled_minority) > 0 and len(undersampled_majority) > 0:
        return pd.concat([oversampled_minority, undersampled_majority])
    elif len(oversampled_minority) > 0:
        return oversampled_minority
    elif len(undersampled_majority) > 0:
        return undersampled_majority
    else:
        return None


def normalize_data_2d(dataframe, column1, column2):
    """
    Function for normalizing data over two columns

    :param dataframe: dataframe to normalize
    :param column1: first column to normalize
    :param column2: second column to normalize
    :return: normalized dataframe
    """

    # get min frequency of a label in first column
    min_value_col1 = min(dataframe[column1].value_counts())

    # normalize data in first column
    first_column = normalize_data(dataframe, column1, min_value_col1)

    splitted = [group_df for _, group_df in first_column.groupby(column1)]
    min_value = min(first_column.groupby(column1)[column2].value_counts())

    # normalize data for second column depending on first column
    result = pd.DataFrame()
    for age_df_split in splitted:
        norm_gen = normalize_data(age_df_split, column2, min_value)
        if result.empty:
            result = norm_gen.copy()
        else:
            result = pd.concat([result, norm_gen], ignore_index=True)

    return result


def map_to_age_group(age):
    """
    Function for converting age to an age group

    :param age: age in years
    :return: age group
    """
    if 10 <= age < 20:
        return '10s'
    elif 20 <= age < 30:
        return '20s'
    # Add more conditions for other age groups as needed
    elif 30 <= age < 50:
        return '30s'
    # Add more conditions for other age groups as needed
    else:
        return 'Unknown'


def import_data(file_path, rows):
    """
    Function for importing data from csv

    :param file_path: path of the csv file
    :param rows: amount of rows to import
    :return: pandas dataframe
    """
    data_frame = pd.read_csv(file_path, delimiter=',', nrows=rows, encoding='utf-8', on_bad_lines='skip')

    return data_frame


def preprocess_data(data_frame):
    """
    Function for preprocessing text column of a dataframe

    :param data_frame: a pandas dataframe
    :return: preprocessed dataframe
    """
    data_frame['text'] = data_frame['text'].apply(clean_text)
    data_frame['age'] = data_frame['age'].apply(map_to_age_group)
    return data_frame


def clean_text(text):
    """
    Function for cleaning text by replacing special characters and stopwords

    :param text: text to be cleaned
    :return: cleaned text
    """

    # Replace all special characters with a whitespace to ensure that words are still split
    # f.e. data point #681283 "Hey everybody...and Susan"
    text = re.sub(r"[^\w\s]", " ", text)

    # Replace all multiple whitespaces with a single whitespace
    text = re.sub(r"^\s+", " ", text.strip()).strip()

    # Remove stopwords
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in STOP_WORDS]
    return ' '.join(tokens)


def preprocess_data_multiprocessing(data_frame):
    """
    Function for preprocessing data on multiple cores
    using amount of available cores - 2

    :param data_frame: a pandas dataframe
    :return: preprocessed dataframe
    """

    # get the amount of available cpus
    cpu_count = mp.cpu_count() - 2

    # split the dataframe into length/cpu_count dataframes
    df_split = split_dataframe_into_chunks(data_frame, cpu_count)

    # df_split = np.array_split(data_frame, cpu_count) - deprecated

    # create pool with cpu_count amount of threads
    with mp.Pool(cpu_count) as p:
        # concat the split dataframes to a result dataframe and apply preprocess_data function on every split dataframe
        df = pd.concat(p.map(preprocess_data, df_split))

    return df


def split_dataframe_into_chunks(data_frame, chunk_size):
    """
    Function for splitting a pandas array into equal chunks
    Used because numpy.array_split() uses deprecated function

    :param data_frame: a pandas dataframe
    :param chunk_size: number of chunks to split the dataframe into
    :return: split dataframe
    """

    # Calculate the number of rows in each part and the remainder
    rows_per_part, remainder = divmod(len(data_frame), chunk_size)

    # Split the DataFrame into equal parts
    df_split = [data_frame.iloc[i * rows_per_part:(i + 1) * rows_per_part] for i in range(chunk_size)]

    # Distribute remaining rows among the parts
    for i in range(remainder):
        df_split[i] = pd.concat(
            [df_split[i], data_frame.iloc[chunk_size * rows_per_part + i:i + 1 + chunk_size * rows_per_part]])

    return df_split


def prepare_data_with_label(data_frame, column):
    """
    Function for preparing data
    by creating a dataframe with text and labels as columns

    :param data_frame: a pandas dataframe
    :param column: the column name with the values
    :return: a dataframe with text and labels
    """
    result_data_frame = data_frame[["text", column]].copy()
    result_data_frame.columns = ['text', 'label']

    result_data_frame["label"] = result_data_frame["label"].astype(str)

    return result_data_frame


def split_training_data(data_frame, test_split_percentage, shuffle_state, shuffle):
    """
    Function for splitting data into training and testing sets

    :param data_frame: a pandas dataframe
    :param test_split_percentage: percentage of the test data
    :param shuffle_state: state to reproducing the shuffle process
    :param shuffle: flag for shuffling
    :return: xtrain, xtest, ytrain, ytest
    """
    x = data_frame["text"]
    y = data_frame["label"]

    return train_test_split(x.values, y.values, test_size=test_split_percentage,
                            random_state=shuffle_state, shuffle=shuffle)
