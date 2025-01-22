import pandas as pd

liar_path = r'Cleaned_Dataset\cleaned_liar_dataset.csv'
kaggle_path = r'Cleaned_Dataset\cleaned_kaggle_train.csv'
github_path = r'Cleaned_Dataset\cleaned_github_dataset.csv'

liar_data = pd.read_csv(liar_path)
kaggle_data = pd.read_csv(kaggle_path)
github_data = pd.read_csv(github_path)

#standardization
liar_data = liar_data.rename(columns={'statement': 'text', 'binary_label': 'label'})
kaggle_data = kaggle_data.rename(columns={'text': 'text', 'label': 'label'})
github_data = github_data.rename(columns={'Post URL': 'text', 'binary_label': 'label'})

#club the dataset
combined_data = pd.concat([liar_data, kaggle_data, github_data], ignore_index=True)

#randomize the rows
combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)

output_path = r'C:\Adi\PROJECTS\Fake News Detection\combined_dataset.csv'
combined_data.to_csv(output_path, index=False)

print(f"Combined dataset saved at {output_path}")
