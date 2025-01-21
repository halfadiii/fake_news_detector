import pandas as pd

# Load the dataset
train_kaggle = pd.read_csv(r'kaggle-fake-news\train.csv', header=0)  # Use header=0 to read column names

# Verify the structure of the dataset
print(train_kaggle.info())  # Ensure 'title', 'text', and 'label' columns exist

# Select relevant columns: 'title', 'text', 'label'
# Ensure columns exist before selection
if {'title', 'text', 'label'}.issubset(train_kaggle.columns):
    train_kaggle_cleaned = train_kaggle[['title', 'text', 'label']]

    # Fill missing text with title if available, otherwise drop the row
    train_kaggle_cleaned['text'] = train_kaggle_cleaned['text'].fillna(train_kaggle_cleaned['title'])
    train_kaggle_cleaned = train_kaggle_cleaned.dropna(subset=['text'])

    # Keep only 'text' and 'label' columns
    train_kaggle_cleaned = train_kaggle_cleaned[['text', 'label']]

    # Save the cleaned dataset
    train_kaggle_cleaned.to_csv(r'C:\Adi\PROJECTS\Fake News Detection\Cleaned_Dataset\cleaned_kaggle_train.csv', index=False)
    print("Cleaning complete. Cleaned Kaggle training dataset saved as 'cleaned_kaggle_train.csv'.")
else:
    print("The dataset does not contain the required columns: 'title', 'text', 'label'.")
