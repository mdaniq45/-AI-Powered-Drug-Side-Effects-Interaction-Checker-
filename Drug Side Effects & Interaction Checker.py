
import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification, pipeline

# Load Hugging Face models
bio_ner_model = "dmis-lab/biobert-base-cased-v1.1"
sci_class_model = "allenai/scibert_scivocab_uncased"
interaction_model = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

# Load tokenizers and models
tokenizer_ner = AutoTokenizer.from_pretrained(bio_ner_model)
model_ner = AutoModelForTokenClassification.from_pretrained(bio_ner_model)
ner_pipeline = pipeline("ner", model=model_ner, tokenizer=tokenizer_ner)

tokenizer_class = AutoTokenizer.from_pretrained(sci_class_model)
model_class = AutoModelForSequenceClassification.from_pretrained(sci_class_model)
class_pipeline = pipeline("text-classification", model=model_class, tokenizer=tokenizer_class)

tokenizer_interaction = AutoTokenizer.from_pretrained(interaction_model)
model_interaction = AutoModelForSequenceClassification.from_pretrained(interaction_model)
interaction_pipeline = pipeline("text-classification", model=model_interaction, tokenizer=tokenizer_interaction)

# Streamlit UI
st.set_page_config(page_title="💊 AI-Powered Drug Side Effects & Interaction Checker", layout="centered")
st.title("💊 Drug Side Effects & Interaction Checker")

# User input
drug_name = st.text_input("Enter a drug name:", "")

if st.button("Analyze Drug"):
    if drug_name:
        # Example text for analysis (Replace with real medical text)
        text = f"{drug_name} is used to treat pain but may cause nausea, dizziness, or headache in some cases."

        # 1️⃣ BioBERT - Named Entity Recognition (Extract drug-related terms)
        ner_results = ner_pipeline(text)
        extracted_terms = [entity['word'] for entity in ner_results if entity['entity'] == 'B-MISC']
        
        # 2️⃣ SciBERT - Side Effect Classification
        class_result = class_pipeline(f"{drug_name} causes dizziness and nausea")
        side_effects = class_result[0]['label']
        
        # 3️⃣ PubMedBERT - Drug Interaction Detection
        interaction_result = interaction_pipeline(f"{drug_name} interacts with aspirin and ibuprofen")
        interaction_label = interaction_result[0]['label']

        # Display Results
        st.subheader(f"🔍 Analysis for **{drug_name}**")
        if extracted_terms:
            st.success(f"📌 **Extracted Medical Terms:** {', '.join(set(extracted_terms))}")
        else:
            st.warning("No significant drug-related terms found.")
        
        st.info(f"💊 **Possible Side Effects:** {side_effects}")
        st.error(f"⚠️ **Drug Interaction Alert:** {interaction_label}")
    else:
        st.error("⚠️ Please enter a drug name.")

# Run with: streamlit run script.py
