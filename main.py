import pandas as pd



def main():
    print("Start")
    import_data(FILE_PATH)


def import_data(file_path):
    data = pd.read_csv(file_path, delimiter=',', nrows=COMPLETE_DATA_LENGTH, encoding='utf-8', on_bad_lines='skip')
    print(data)


FILE_PATH = "assets/blogtext.csv"
COMPLETE_DATA_LENGTH = 681284

# driver
if __name__ == "__main__":
    main()
