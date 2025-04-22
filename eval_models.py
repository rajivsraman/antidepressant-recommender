import os
import openai
import joblib
import torch
import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from dotenv import load_dotenv
from load_data import DataLoader

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load models and tokenizer
def load_models():
    classical_model, tfidf = joblib.load("models/classical_model.pkl")
    deep_model = DistilBertForSequenceClassification.from_pretrained("models/deep_model")
    tokenizer = DistilBertTokenizerFast.from_pretrained("models/deep_model")
    deep_model.eval()
    return classical_model, tfidf, deep_model, tokenizer

# Load filtered dataset and extract antidepressants
def load_depression_data():
    df = DataLoader().load_filtered_data()
    df = df[df["condition"].fillna("").str.lower().str.contains("depress")]
    return df, sorted(df["urlDrugName"].unique())

# Generate 50 synthetic patient prompts
def generate_patient_prompts():
    system_prompt = (
        "You are simulating realistic patient symptom descriptions for depression."
        " Generate 50 varied and detailed prompts, each 1-3 sentences long, describing"
        " how a patient might communicate their symptoms. Each prompt should include"
        " specific emotional, physical, and behavioral details. Return them as a numbered list."
    )
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system_prompt}],
        temperature=0.7
    )
    raw_output = response.choices[0].message.content
    return [line.partition(". ")[2].strip() for line in raw_output.strip().split("\n") if ". " in line][:50]

# Generate LLM summary
def generate_llm_summary(prompt, top_drugs):
    summary_prompt = (
        f"A patient describes their symptoms: {prompt}\n\n"
        f"Top 5 recommended antidepressants: {', '.join(top_drugs)}\n\n"
        "Please explain in one paragraph why these drugs might be appropriate choices for the described symptoms."
    )
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": summary_prompt}],
            temperature=0.6
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[LLM summary failed: {str(e)}]"

# Score the summary using LLM
def score_summary_quality(prompt, summary):
    scoring_prompt = (
        f"A patient describes their symptoms: {prompt}\n\n"
        f"Here is a machine-generated summary of antidepressant recommendations:\n{summary}\n\n"
        "Rate this summary on a scale from 1 to 5 based on its relevance and helpfulness in addressing the patient's symptoms. Make sure there is a good deal of variance in your responses.\n"
        "Respond with a number only."
    )
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": scoring_prompt}],
            temperature=0.3
        )
        score = response.choices[0].message.content.strip()
        return int(score) if score.isdigit() else score
    except Exception as e:
        return f"[Scoring failed: {str(e)}]"

# Run full evaluation pipeline
def run_pipeline():
    classical_model, tfidf, deep_model, deep_tokenizer = load_models()
    df, all_drugs = load_depression_data()
    prompts = generate_patient_prompts()
    results = []

    for i, prompt in enumerate(prompts):
        print(f"\nProcessing prompt {i+1}/{len(prompts)}")
        classical_scores = {}
        deep_scores = {}

        for j, drug in enumerate(all_drugs):
            if j % 25 == 0:
                print(f"   → Ranking drug {j+1}/{len(all_drugs)}")

            row = df[df["urlDrugName"] == drug].iloc[0]
            structured_context = f"{row['effectiveness']} {row['sideEffects']} {row['condition']}"
            input_text = f"{prompt} {structured_context} {drug}"

            X_vec = tfidf.transform([input_text])
            classical_rating = classical_model.predict(X_vec)[0]
            classical_scores[drug] = classical_rating

            inputs = deep_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                logits = deep_model(**inputs).logits
            predicted_rating = torch.argmax(logits, dim=1).item() + 1
            deep_scores[drug] = predicted_rating

        top_classical = sorted(classical_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        top_deep = sorted(deep_scores.items(), key=lambda x: x[1], reverse=True)[:5]

        print("Summaries generating...")
        summary_classical = generate_llm_summary(prompt, [name for name, _ in top_classical])
        summary_deep = generate_llm_summary(prompt, [name for name, _ in top_deep])

        print("Scoring summaries...")
        score_classical = score_summary_quality(prompt, summary_classical)
        score_deep = score_summary_quality(prompt, summary_deep)

        results.append({
            "prompt": prompt,
            "top_classical": [name for name, _ in top_classical],
            "top_deep": [name for name, _ in top_deep],
            "summary_classical": summary_classical,
            "summary_deep": summary_deep,
            "score_classical": score_classical,
            "score_deep": score_deep
        })

    df_out = pd.DataFrame(results)
    df_out.to_csv("model_summary_scores.csv", index=False)

    valid_classical_scores = [r["score_classical"] for r in results if isinstance(r["score_classical"], int)]
    valid_deep_scores = [r["score_deep"] for r in results if isinstance(r["score_deep"], int)]

    avg_classical = sum(valid_classical_scores) / len(valid_classical_scores) if valid_classical_scores else 0
    avg_deep = sum(valid_deep_scores) / len(valid_deep_scores) if valid_deep_scores else 0

    print(f"\nSaved to outputs/model_summary_scores.csv")
    print(f"Average Classical Summary Score: {avg_classical:.2f}")
    print(f"Average Deep Summary Score: {avg_deep:.2f}")


if __name__ == "__main__":
    run_pipeline()

