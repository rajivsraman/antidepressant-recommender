import pandas as pd
from ucimlrepo import fetch_ucirepo

class DataLoader:
    """
    Loads and filters the Drug Reviews dataset from UCI for a specified condition.
    """

    def __init__(self, condition_keyword="depression"):
        """
        Initializes the DataLoader with a condition keyword for filtering.

        Args:
            condition_keyword (str): Condition to filter the dataset on (default is 'depression').
        """
        self.condition_keyword = condition_keyword.lower()

    def load_filtered_data(self):
        """
        Loads and returns a filtered DataFrame containing only rows relevant to the specified condition.

        Returns:
            pd.DataFrame: Filtered dataset.
        """
        dataset = fetch_ucirepo(id=461)
        df = dataset.data.features
        df = df[df["condition"].str.lower().str.contains(self.condition_keyword, na=False)]
        df = df.dropna(subset=["rating", "benefitsReview"])
        df["rating"] = df["rating"].astype(int)
        return df