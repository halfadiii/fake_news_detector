import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

# Load the SpaCy model for preprocessing
nlp = spacy.load("en_core_web_sm")

# Function to preprocess text
def preprocess_text(doc):
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

# Load the saved TF-IDF vectorizer and logistic regression model
# Replace 'path_to_vectorizer' and 'path_to_model' with the actual paths to your .pkl files
vectorizer_path = r'C:\Adi\PROJECTS\Fake News Detection\Full Trained Models\tfidf_vectorizer.pkl'
model_path = r'C:\Adi\PROJECTS\Fake News Detection\Full Trained Models\rf_fake_news_model.pkl'

with open(vectorizer_path, 'rb') as file:
    vectorizer = pickle.load(file)
with open(model_path, 'rb') as file:
    model = pickle.load(file)

#to predict
news_articles = [
    "The moon is made of cheese.",
    "Commerce Secretary nominee John Bryson appears to endorse world government.",
    "This is the first time in Texas history -- and only the fourth time in United States history -- that two women are running for the top spots.",
    "Rep. Ron DeSantis ( ) talked with Breitbart News Daily SiriusXM host Raheem Kassam on Thursday regarding his experience at the GOP baseball practice where Rep. Steve Scalise was shot. [DeSantis said while Rep. Scalise â€œis fighting for his life, I think heâ€™s going to pull through. â€  After describing his interactions with the alleged shooter, DeSantis said, â€œWe received a message expressing approval of what had happened and just hoping that Donald Trump would be next. And another one of my colleagues received an email saying one down, 217 more to go. and other colleagues have received other things. â€ DeSantis said he believes the hate being engendered toward President Trump is being directed at Congress because they are more accessible. â€œThis guy went there clearly filled with rage, clearly filled with political ideology that was hostile to the president and Republicans, and he wanted to kill a lot of Republicans. â€ Breitbart News Daily airs on SiriusXM Patriot 125 weekdays from 6:00 a. m. to 9:00 a. m. Eastern. LISTEN: ",
    "Today, there are more Hoosiers going to work than ever before in the 200-year history of the great state of Indiana."
]

# Preprocess and vectorize the news articles
processed_articles = [preprocess_text(nlp(article)) for article in news_articles]
article_vectors = vectorizer.transform(processed_articles)

# Predict the labels of the news articles
predictions = model.predict(article_vectors)

# Output the predictions
for article, prediction in zip(news_articles, predictions):
    print(f"Article: {article}\nPrediction: {'Real' if prediction == 1 else 'Fake'}\n")
