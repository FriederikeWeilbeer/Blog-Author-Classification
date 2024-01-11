import pandas as pd


def main():
    print("Start")
    import_data(FILE_PATH)


def import_data(file_path):
    data = pd.read_csv(file_path, delimiter=',', nrows = 5000, engine='python', encoding='utf-8')


FILE_PATH = "assets/blogtext.csv"

# driver
if __name__ == "__main__":
    main()
