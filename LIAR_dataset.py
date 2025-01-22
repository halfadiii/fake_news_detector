import pandas as pd

# Load the dataset
liar_data = pd.read_csv(r'Dataset\train.tsv', sep='\t', header=None)

# Add column names
columns = [
    'id', 'label', 'statement', 'subjects', 'speaker',
    'speaker_job', 'state_info', 'party', 'barely_true_count',
    'false_count', 'half_true_count', 'mostly_true_count',
    'pants_on_fire_count', 'context'
]
liar_data.columns = columns

# Handle missing values: Drop rows where 'statement' or 'label' is missing
liar_data = liar_data.dropna(subset=['statement', 'label'])

# Normalize labels into binary: Fake (0), Real (1)
label_mapping = {
    'pants-fire': 0, 'false': 0, 'barely-true': 0,
    'half-true': 1, 'mostly-true': 1, 'true': 1
}
liar_data['binary_label'] = liar_data['label'].map(label_mapping)

# Keep only relevant columns
cleaned_liar_data = liar_data[['statement', 'binary_label']]

# Save the cleaned dataset
cleaned_liar_data.to_csv(r'C:\Adi\PROJECTS\Fake News Detection\Cleaned_Dataset\cleaned_liar_dataset.csv', index=False)

print("Cleaning complete. Cleaned dataset saved as 'cleaned_liar_train.csv'.")
