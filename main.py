import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
import time
from tqdm import tqdm


def main():
    print("Start")
    now = time.time()
    data = import_data(FILE_PATH, 100000)
    print(data.columns)
    prepare_data(data)
    print(time.time() - now)


def import_data(file_path, rows):
    data_frame = pd.read_csv(file_path, delimiter=',', nrows=rows, encoding='utf-8', on_bad_lines='skip')
    return data_frame


def remove_stopwords(text):

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]

    return ' '.join(tokens)


def prepare_data(data_frame):
    print(data_frame['text'].apply(len).mean())
    data_frame['text'] = data_frame['text'].apply(remove_stopwords)
    print(data_frame['text'].apply(len).mean())

FILE_PATH = "assets/blogtext.csv"
COMPLETE_DATA_LENGTH = 681284

# driver
if __name__ == "__main__":
    main()
