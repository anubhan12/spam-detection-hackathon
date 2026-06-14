import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from preprocess import TextPreprocessor


def run_training_pipeline(data_path: str, model_dir: str = "models"):
    os.makedirs(model_dir, exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(data_path, encoding='latin-1')

    # Clean dataset format
    df = df[['v1', 'v2']].copy()
    df.columns = ['label', 'text']

    df = df.dropna(subset=['label', 'text']).reset_index(drop=True)
    df['text'] = df['text'].astype(str)

    preprocessor = TextPreprocessor()

    print("Preprocessing text...")
    df['cleaned_text'] = df['text'].apply(lambda x: preprocessor.clean_text(x))

    print("Sanity check:", df.shape)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'],
        df['label'],
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )

    print("Vectorizing data...")

    # 🚀 IMPROVED TF-IDF (BIG UPGRADE)
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training Logistic Regression model...")

    # 🚀 STRONGER MODEL (KEY IMPROVEMENT)
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    model.fit(X_train_vec, y_train)

    print("\n--- Evaluation ---")

    predictions = model.predict(X_test_vec)

    print(classification_report(y_test, predictions))

    print("Saving model...")

    joblib.dump(model, os.path.join(model_dir, "model.pkl"))
    joblib.dump(vectorizer, os.path.join(model_dir, "vectorizer.pkl"))

    print("Training complete!")


if __name__ == "__main__":
    run_training_pipeline("data/raw/spam.csv")