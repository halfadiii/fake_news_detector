import pandas as pd
github_data = pd.read_csv(r'C:\Adi\PROJECTS\Fake News Detection\Github Data\facebook-fact-check.csv')

## use post URL and Rating
github_cleaned = github_data[['Post URL', 'Rating']]

#change rating column to binary
rating_mapping = {
    'no factual content': 0,
    'mostly false': 0,
    'false': 0,
    'mostly true': 1,
    'true': 1
}
github_cleaned['binary_label'] = github_cleaned['Rating'].map(rating_mapping)

#Drop rows 
github_cleaned = github_cleaned.dropna(subset=['binary_label'])

## use post URL and Rating
github_cleaned = github_cleaned[['Post URL', 'binary_label']]

output_path = r'C:\Adi\PROJECTS\Fake News Detection\Cleaned_Dataset\cleaned_github_dataset.csv'
github_cleaned.to_csv(output_path, index=False)

print(f"Cleaning complete. Cleaned GitHub dataset saved at '{output_path}'.")
