import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from load_data import DataLoader

class ClassicalTrainer:
    """
    Trains a TF-IDF + Linear Regression model using depression-specific drug review text and emphasized fields.
    """

    def __init__(self, data):
        # Filter for depression only
        self.data = data[data["condition"].str.lower().str.contains("depress", na=False)].copy()
        print(f"Depression-specific entries: {len(self.data)}")
        print(f"Unique drugs: {self.data['urlDrugName'].nunique()}")

    def emphasize_effectiveness(self, row):
        return f"This drug was reported as '{row['effectiveness']}' for treating {row['condition']}."

    def prepare_input_text(self):
        return (
            self.data.apply(self.emphasize_effectiveness, axis=1) + " " +
            self.data["benefitsReview"].fillna("") + " " +
            self.data["sideEffectsReview"].fillna("") + " " +
            self.data["commentsReview"].fillna("") + " " +
            self.data["sideEffects"].fillna("")
        )

    def train(self, save_path="models/classical_model.pkl"):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        self.data = self.data.dropna(subset=["rating", "benefitsReview", "effectiveness", "condition"])
        self.data = self.data[self.data["rating"].astype(str).str.isnumeric()]
        X = self.prepare_input_text()
        y = self.data["rating"].astype(int)

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

        tfidf = TfidfVectorizer(max_features=3000, stop_words="english")
        X_train_vec = tfidf.fit_transform(X_train)

        model = LinearRegression()
        model.fit(X_train_vec, y_train)

        joblib.dump((model, tfidf), save_path)
        print("Classical model trained and saved.")

if __name__ == "__main__":
    trainer = ClassicalTrainer(DataLoader().load_filtered_data())
    trainer.train()
