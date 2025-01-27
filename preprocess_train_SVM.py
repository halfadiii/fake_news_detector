import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import pickle
from tqdm import tqdm

dataset_path = r'combined_dataset.csv'
data = pd.read_csv(dataset_path)
data = data.head(37394)
nlp = spacy.load("en_core_web_sm")

def preprocess_text(doc):
    #tokenize, drop stop words, punctuation, and lemmatize
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

# Apply preprocessing to text data
data['processed_text'] = [preprocess_text(nlp(text)) for text in tqdm(data['text'], desc="Preprocessing Text")]

#tf-idf
vectorizer = TfidfVectorizer(max_features=37394)
X = vectorizer.fit_transform(data['processed_text'])
y = data['label']

#80:20 train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the Model (Support Vector Machine)
print("Starting SVM training...")
model = SVC(kernel='linear', verbose=True)  # Using linear kernel for simplicity
model.fit(X_train, y_train)

# Step 7: Evaluate the Model
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 8: Save the Model and Vectorizer
with open('svm_fake_news_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("SVM model and vectorizer saved successfully!")
