import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from load_data import DataLoader

class DeepTrainer:
    """
    Trains a DistilBERT model using enriched inputs that emphasize drug effectiveness
    and organize review sections into token-tagged segments for better contextual learning.
    """

    def __init__(self, data):
        """
        Initializes the trainer by filtering for depression-related records and setting the computation device.

        Args:
            data (pd.DataFrame): The full drug review dataset.
        """
        self.data = data[data["condition"].str.lower().str.contains("depress", na=False)].copy()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Depression-specific entries: {len(self.data)}")
        print(f"Unique drugs: {self.data['urlDrugName'].nunique()}")
        print(f"Using device: {self.device}")

    def format_for_bert(self, row):
        """
        Creates a token-tagged string combining structured fields and reviews for BERT input.

        Args:
            row (pd.Series): A row from the dataset.

        Returns:
            str: Formatted input text string for the model.
        """
        effectiveness_line = f"[EFFECTIVENESS] {row['effectiveness']} [/EFFECTIVENESS]"
        condition_line = f"[CONDITION] {row['condition']} [/CONDITION]"
        side_effects_line = f"[SIDE_EFFECTS] {row['sideEffects']} [/SIDE_EFFECTS]"
        benefits = row.get("benefitsReview", "") or ""
        side_effects = row.get("sideEffectsReview", "") or ""
        comments = row.get("commentsReview", "") or ""
        review_text = f"[BENEFITS] {benefits} [/BENEFITS] [SIDE_REVIEW] {side_effects} [/SIDE_REVIEW] [COMMENTS] {comments} [/COMMENTS]"
        return f"{effectiveness_line} {condition_line} {side_effects_line} {review_text}"

    def preprocess(self):
        """
        Prepares the tokenized train and validation datasets using structured and review text.

        Returns:
            tuple: Tokenized HuggingFace Datasets and tokenizer.
        """
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

        self.data = self.data.dropna(subset=["rating", "effectiveness", "condition", "benefitsReview"])
        self.data = self.data[self.data["rating"].astype(str).str.isnumeric()]
        self.data["label"] = self.data["rating"].astype(int) - 1
        self.data["text"] = self.data.apply(self.format_for_bert, axis=1)

        train_df, val_df = train_test_split(self.data, test_size=0.2, stratify=self.data["label"], random_state=42)

        train_dataset = Dataset.from_pandas(train_df[["text", "label"]])
        val_dataset = Dataset.from_pandas(val_df[["text", "label"]])

        tokenize_fn = lambda x: tokenizer(x["text"], padding=True, truncation=True)
        train_dataset = train_dataset.map(tokenize_fn, batched=True)
        val_dataset = val_dataset.map(tokenize_fn, batched=True)

        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        return train_dataset, val_dataset, tokenizer

    def train(self, model_dir="models/deep_model"):
        """
        Trains the DistilBERT model and saves both model and tokenizer to disk.

        Args:
            model_dir (str): Path to directory where model files should be saved.
        """
        os.makedirs(model_dir, exist_ok=True)
        train_dataset, val_dataset, tokenizer = self.preprocess()

        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=10
        ).to(self.device)

        training_args = TrainingArguments(
            output_dir=model_dir,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=10,
            logging_dir="logs",
            report_to="none",
            save_total_limit=1,
            save_strategy="no",
            logging_strategy="epoch"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        trainer.train()
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        print(f"Deep model trained and saved to {model_dir}")

if __name__ == "__main__":
    trainer = DeepTrainer(DataLoader().load_filtered_data())
    trainer.train()