import streamlit as st
import joblib
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, PretrainedConfig, PreTrainedTokenizerFast
import tempfile
from scripts.load_data import DataLoader
import os
from dotenv import load_dotenv
from openai import OpenAI

os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"

load_dotenv()

class CDSSApp:
    """
    CDSS application that ranks top 5 antidepressants based on naive, classical, or deep models.
    """

    def __init__(self):
        self.symptoms = ""
        self.model_choice = ""
        self.use_llm = False
        self.df = DataLoader().load_filtered_data()
        self.df = self.df[self.df["condition"].str.lower().str.contains("depress", na=False)]
        self.all_drugs = sorted(self.df["urlDrugName"].unique())

    def sidebar(self):
        st.sidebar.title("Model Selection")
        self.model_choice = st.sidebar.radio("Select Model", ["Naive", "TF-IDF", "DistilBERT"])
        if self.model_choice != "Naive":
            self.symptoms = st.sidebar.text_area("Describe symptoms")
            self.use_llm = st.sidebar.checkbox("Summarize results with LLM (OpenAI)")

    def load_naive_model(self):
        return joblib.load("models/naive_model.pkl")

    def load_classical_model(self):
        return joblib.load("models/classical_model.pkl")

    def load_deep_model(self):

    gcs_base = "https://storage.googleapis.com/adrs-distilbert/deep_model"
    required_files = [
        "model.safetensors",
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.txt",
        "special_tokens_map.json"
    ]

    temp_dir = tempfile.mkdtemp()

    # Download files directly into temp folder
    for fname in required_files:
        url = f"{gcs_base}/{fname}"
        local_path = os.path.join(temp_dir, fname)
        print(f"Downloading {fname}")
        result = os.system(f"curl -f -s {url} -o {local_path}")
        if result != 0:
            raise RuntimeError(f"Failed to download: {url}")

    # Load directly from temp folder
    model = DistilBertForSequenceClassification.from_pretrained(temp_dir)
    tokenizer = DistilBertTokenizerFast.from_pretrained(temp_dir)

    return model, tokenizer


    def summarize_results_with_llm(self, top_5):
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        prompt = """
        You are a helpful medical assistant. A patient has described the following symptoms:
        """
        prompt += f"{self.symptoms}\n\n"
        prompt += "Based on a machine learning model, here are the top 5 predicted antidepressants:\n"
        for i, (name, _) in enumerate(top_5, 1):
            prompt += f"{i}. {name}\n"

        prompt += "\nPlease determine which of the five options are the best suited as an antidepressant based on the provided symptoms. When you determine the best option, please give a summary explaining why this was your choice. Do not discuss the other four options. Explain the benefits along with the potential side effects. If none of the five options are good for this situation, please explicitly state that and tell the user that you will be unhelpful.\n"

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[LLM summary failed: {str(e)}]"

    def run_naive(self):
        st.markdown("### Recommended Antidepressants (Naive Baseline Model)")
        st.info("This model ranks drugs based on overall patient satisfaction and number of reviews. No user input is required.")
        ratings_dict = self.load_naive_model()
        top_5 = sorted(ratings_dict.items(), key=lambda x: x[1]["combined_score"], reverse=True)[:5]
        st.table({"Recommended Antidepressants": [name for name, _ in top_5]})

    def run_classical(self):
        st.markdown("### Recommended Antidepressants (TF-IDF Model)")
        if not self.symptoms.strip():
            st.warning("Please enter symptoms in the sidebar.")
            return

        model, tfidf = self.load_classical_model()
        scores = {}
        for drug in self.all_drugs:
            row = self.df[self.df["urlDrugName"] == drug].iloc[0]
            structured_context = f"{row['effectiveness']} {row['sideEffects']} {row['condition']}"
            input_text = f"{self.symptoms} {structured_context} {drug}"
            X_vec = tfidf.transform([input_text])
            pred = model.predict(X_vec)[0]
            scores[drug] = pred

        top_5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        st.table({"Recommended Antidepressants": [d[0] for d in top_5]})

        if self.use_llm:
            st.markdown("#### Summary:")
            llm_output = self.summarize_results_with_llm(top_5)
            st.write(llm_output)

    def run_deep(self):
        st.markdown("### Recommended Antidepressants (DistilBERT Model)")
        if not self.symptoms.strip():
            st.warning("Please enter symptoms in the sidebar.")
            return

        model, tokenizer = self.load_deep_model()
        model.eval()
        scores = {}
        for drug in self.all_drugs:
            row = self.df[self.df["urlDrugName"] == drug].iloc[0]
            structured_context = f"{row['effectiveness']} {row['sideEffects']} {row['condition']}"
            input_text = f"{self.symptoms} {structured_context} {drug}"
            inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                logits = model(**inputs).logits
            pred_rating = torch.argmax(logits, dim=1).item() + 1
            scores[drug] = pred_rating

        top_5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        st.table({"Recommended Antidepressants": [d[0] for d in top_5]})

        if self.use_llm:
            st.markdown("#### Summary:")
            llm_output = self.summarize_results_with_llm(top_5)
            st.write(llm_output)

    def run(self):
        st.title("Antidepressant Recommendation System")
        self.sidebar()

        if self.model_choice == "Naive":
            self.run_naive()
        elif self.model_choice == "TF-IDF":
            self.run_classical()
        elif self.model_choice == "DistilBERT":
            self.run_deep()

if __name__ == '__main__':
    app = CDSSApp()
    app.run()
