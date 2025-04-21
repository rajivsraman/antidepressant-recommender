import os
import joblib
import pandas as pd
from math import log10
from load_data import DataLoader

class NaiveTrainer:
    """
    Computes and saves a combined score for each drug based on average rating and number of reviews.
    Also prints the top 10 drugs by combined score.
    """

    def __init__(self, data):
        """
        Initializes the trainer with dataset containing 'urlDrugName' and 'rating'.

        Args:
            data (pd.DataFrame): Filtered dataset.
        """
        self.data = data

    def compute_combined_score(self, row):
        """
        Computes a combined score using rating and log-scaled review count.

        Args:
            row (pd.Series): Row with 'average_rating' and 'num_reviews'.

        Returns:
            float: Combined score.
        """
        return row["average_rating"] * log10(row["num_reviews"] + 1)

    def train(self, save_path="models/naive_model.pkl"):
        """
        Computes average rating, review count, and combined score for each drug.
        Saves full dictionary and prints top 10 by combined score.

        Args:
            save_path (str): Path to save the model dictionary.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Compute average and count
        grouped = self.data.groupby("urlDrugName")["rating"].agg(["mean", "count"])
        grouped = grouped.rename(columns={"mean": "average_rating", "count": "num_reviews"})

        # Add combined score
        grouped["combined_score"] = grouped.apply(self.compute_combined_score, axis=1)

        # Save as dictionary
        model_dict = grouped.to_dict(orient="index")
        joblib.dump(model_dict, save_path)

        # Show top 10 by combined score
        top_10 = grouped.sort_values(by="combined_score", ascending=False).head(10)
        print("Top 10 drugs by combined score (rating × log10(review_count + 1)):")
        print(top_10)

if __name__ == '__main__':
    df = DataLoader().load_filtered_data()
    trainer = NaiveTrainer(df)
    trainer.train()
