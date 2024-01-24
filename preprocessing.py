import pandas as pd
from sklearn.utils import resample
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

from main import import_data


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
    min_value_col1 = min(dataframe[column1].value_counts())
    first_column = normalize_data(dataframe, column1, min_value_col1)

    splitted = [group_df for _, group_df in first_column.groupby(column1)]
    min_value = min(first_column.groupby(column1)[column2].value_counts())

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
    """
    if 10 <= age < 20:
        return '10s'
    elif 20 <= age < 30:
        return '20s'
    # Add more conditions for other age groups as needed
    elif 30 <= age < 40:
        return '30s'
    elif 40 <= age < 50:
        return '40s'
    elif 50 <= age < 60:
        return '50s'
    elif 60 <= age < 70:
        return '60s'
    # Add more conditions for other age groups as needed
    else:
        return 'Unknown'


def main():
    df = import_data(FILE_PATH, COMPLETE_DATA_LENGTH)

    df['age_group'] = df['age'].apply(map_to_age_group)

    data = normalize_data_2d(df, 'age_group', 'gender')



    # Step 1: Prepare Your Dataset
    # Assuming you have a dataset with 'text' and 'age' columns

    # Step 2: Preprocess the Data
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    class AgeDataset(Dataset):
        def __init__(self, texts, ages):
            self.texts = texts
            self.ages = ages

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            age = self.ages[idx]
            return {'text': text, 'age': age}

    # Example usage:
    train_dataset = AgeDataset(df["text"], df["age_group"])
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Step 3: Load Pre-trained BERT Model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)

    # Step 4: Define a Classifier Head
    # Already done by using BertForSequenceClassification

    # Step 5: Fine-tune the Model
    optimizer = AdamW(model.parameters(), lr=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    num_epochs = 10

    # Training loop
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True)
            labels = batch['age']

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # Step 6: Evaluate the Model
    # Evaluate on a validation set

    # Step 7: Make Predictions
    # Use the trained model to make predictions on new text samples


    # print(data.groupby("age_group")["gender"].value_counts().unstack(fill_value=0))




FILE_PATH = "assets/blogtext.csv"
COMPLETE_DATA_LENGTH = 681284
RANDOM_STATE = 41236451
LOGGING = True


if __name__ == '__main__':
    main()
