import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

#SpaCy model
nlp = spacy.load("en_core_web_sm")

def preprocess_text(doc):
    """Normalize, remove stopwords and punctuation, and lemmatize the text."""
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

def check_known_false_claims(text):
    """Check the text for known false claims and adjust predictions accordingly."""
    false_claims = {
        "the moon is made of cheese": False,
        "the earth is flat": False,
        "the earth is square": False
    }
    for claim, is_true in false_claims.items():
        if claim in text.lower():
            return is_true
    return None  #nofalse claim detected

#TF-IDF vectorizer and LR model
vectorizer_path = r'C:\Adi\PROJECTS\Fake News Detection\Full Trained Models\tfidf_vectorizer.pkl'
model_path = r'C:\Adi\PROJECTS\Fake News Detection\Full Trained Models\rf_fake_news_model.pkl'

with open(vectorizer_path, 'rb') as file:
    vectorizer = pickle.load(file)
with open(model_path, 'rb') as file:
    model = pickle.load(file)

news_articles = [
    "The moon is made of cheese.",
    "The earth is square.",
    "Commerce Secretary nominee John Bryson appears to endorse world government.",
    "This is the first time in Texas history -- and only the fourth time in United States history -- that two women are running for the top spots.",
    "Rep. Ron DeSantis talked about the GOP baseball practice incident.",
    "Today, there are more Hoosiers going to work than ever before in the 200-year history of the great state of Indiana."
]

#preprocess and vectorization
processed_articles = [preprocess_text(nlp(article)) for article in news_articles]
article_vectors = vectorizer.transform(processed_articles)

#Predict the labels
predictions = model.predict(article_vectors)

# adjusting predictions (fact check vs Model)
for article, prediction in zip(news_articles,predictions):
    fact_check_result = check_known_false_claims(article)
    if fact_check_result is False:
        prediction=0
    print(f"Article:{article}\nPrediction:{'Real' if prediction==1 else 'Fake'}\n")
