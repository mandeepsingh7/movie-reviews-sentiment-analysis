import re 
import joblib 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from datasets import load_dataset

def clean_text(text):
    # remove HTML line breaks 
    return re.sub(r"<br\s*/?>", " ", text)

def load_ml_data(test_size=0.2, seed=42):
    dataset = load_dataset("imdb", split='train')

    texts = dataset["text"]
    labels = dataset["label"]

    # clean text
    texts = [clean_text(x) for x in texts]
    labels = list(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=seed
    )

    return X_train, X_test, y_train, y_test

def get_tfidf():
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2), # good, not good
        max_features=50000, # 50000 most important features 
        min_df = 5, # ignore words appearing in less than 5 reviews, can be names, mistakes 
        max_df = 0.9, # ignoring too frequent words
        stop_words="english" # ignoring stop words like the, are, am etc. 
    )
    return tfidf

def build_pipeline(model):
    return Pipeline([
        ('tfidf', get_tfidf()),
        ('clf', model)
    ])

def train_and_evaluate(pipeline, X_train, y_train, X_test, y_test):
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy, pipeline  

def save_model(pipeline, filepath):
    joblib.dump(pipeline, filepath)