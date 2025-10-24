import re
import torch
import spacy
import langdetect
from langdetect import detect, DetectorFactory
from googletrans import Translator
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import numpy as np

nlp = spacy.load("en_core_web_sm")

translator = Translator()

MODEL_PATH = r"bert model"
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
bert_model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)
bert_model.eval()

LABELS = {0: "Name", 1: "Job Title", 2: "Company", 3: "Address"}

UNIQUE_CATEGORIES = ["Name", "Job Title", "Company", "Address"]
MULTIPLE_CATEGORIES = ["Phone Number", "Email", "Website"]

# ------------------------ LAYER 1: REGEX-BASED CLASSIFICATION ------------------------
def classify_by_regex(text):
    """Classifies input based on regex rules and applies elimination logic."""
    
    phone_pattern = r"\+?\d{1,3}[-.\s]?\(?\d{2,4}\)?([-.\s]?\d{2,4}){2,4}"
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    website_pattern = r"\b(?:https?:\/\/)?(?:www\.)?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}(?:\.[a-zA-Z]{2,6})?\b"

    company_suffixes = [
        "Inc", "Ltd", "LLC", "Corp", "GmbH", "S.A.", "Pvt", "Co.", "Limited",
        "AG", "PLC", "S.A.S.", "BV", "S.p.A.", "AB", "Oy", "ApS", "NV", "KG",
        "KGaA", "SARL", "LDA", "OÜ", "AS", "JSC", "KK", "PT", "Tbk"
    ]

    address_keywords = [
        "Street", "St.", "Ave", "Avenue", "Road", "Rd.", "Blvd", "Boulevard", "Lane", "Ln.", 
        "Drive", "Dr.", "Square", "Sq.", "Plaza", "Plz.", "Way", "Highway", "Hwy.", 
        "Parkway", "Pkwy.", "Court", "Ct.", "Circle", "Cir.", "District", "Dstr.", "Neighborhood", "Nbh."
    ]

    classifications = {}
    eliminate_name = False

    # ------------------------ Exact Match Classifications ------------------------
    if re.fullmatch(phone_pattern, text):
        classifications["Phone Number"] = ("Regex", 1.0)
    elif re.fullmatch(email_pattern, text):
        classifications["Email"] = ("Regex", 1.0)
    elif re.fullmatch(website_pattern, text):
        classifications["Website"] = ("Regex", 1.0)
    elif any(text.strip().endswith(suffix) for suffix in company_suffixes):
        classifications["Company"] = ("Regex", 0.95)
    elif any(keyword.lower() in text.lower() for keyword in address_keywords):
        classifications["Address"] = ("Regex", 0.90)
    else:
        # ------------------------ Smart Rule-Based Classification ------------------------
        word_count = len(text.split())  
        contains_comma = "," in text  
        contains_digits = any(char.isdigit() for char in text)

        # If the line contains a comma or numbers, eliminate Name classification
        if contains_comma or contains_digits:
            eliminate_name = True

    return classifications, eliminate_name

# ------------------------ LAYER 2: NAMED ENTITY RECOGNITION (NER) ------------------------
def classify_by_ner(text):
    """Uses a pre-trained NER model (spaCy) to classify Name, Company, Address."""
    doc = nlp(text)
    ner_confidences = {category: ("NER", 0.0) for category in LABELS.values()}

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            ner_confidences["Name"] = ("NER", 0.9)
        elif ent.label_ == "ORG":
            ner_confidences["Company"] = ("NER", 0.85)
        elif ent.label_ == "GPE":
            ner_confidences["Address"] = ("NER", 0.80)

    return ner_confidences

# ------------------------ LAYER 3: FINE-TUNED BERT CLASSIFIER WITH CONFIDENCE ------------------------
def classify_by_bert(text):
    """Uses the fine-tuned BERT model to classify the input with confidence scores."""
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = bert_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        probs = probs.cpu().numpy().flatten()

    return {LABELS[i]: ("BERT", probs[i]) for i in range(len(LABELS))}

# ------------------------ MAIN FUNCTION: PROCESS FULL TEXT ------------------------
def classify_text(text):
    """Processes multi-line input, stores confidence scores, and selects best match per category."""

    results = []
    assigned_entries = set()
    processed_lines = set()

    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        if not line or line in processed_lines:
            continue

        processed_lines.add(line)
        print(f"\nProcessing Line: {line}")

        regex_results, eliminate_name = classify_by_regex(line)

        # Attempt translation with retry logic
        translated_line = line  # Default to original
    
        try:
            translated_line = translator.translate(line, src="auto", dest="en").text
            print(f"Translated '{line}' -> '{translated_line}'")
        except Exception as e:
            print(f"Translation failed for '{line}'. Error: {e}")

        if regex_results:
            category, (source, confidence) = list(regex_results.items())[0]
            print(f"[{source}] Classified as {category} (Confidence: {confidence:.2f})")
            results.append((line, translated_line, regex_results, eliminate_name))
            assigned_entries.add(line)
            continue  

        confidence_scores = classify_by_ner(translated_line)
        bert_results = classify_by_bert(translated_line)  

        for category, (source, score) in bert_results.items():
            current_source, current_conf = confidence_scores.get(category, ("NER", 0.0))
            if score > current_conf:
                confidence_scores[category] = ("BERT", score)

        # Adjust Name Confidence if Eliminated
        if eliminate_name:
            confidence_scores["Name"] = ("Filtered", 0.0)  

        print(f"✅ Confidence Scores: {confidence_scores}")
        results.append((line, translated_line, confidence_scores, eliminate_name))

    # ------------------------ FINAL CLASSIFICATION ------------------------
    final_result = {cat: [] for cat in MULTIPLE_CATEGORIES}  
    unique_category_assignments = {}

    # Regex-classified entries are assigned first
    for original_text, _, confidences, _ in results:
        for category in MULTIPLE_CATEGORIES:
            if category in confidences and confidences[category][1] == 1.0:
                final_result[category].append(original_text)
                assigned_entries.add(original_text)

        # Assign Address separately
        if "Address" in confidences and confidences["Address"][1] > 0.8:
            if "Address" in final_result and final_result["Address"]:  
                print(f"⚠️ WARNING: Address already assigned ({final_result['Address']}), skipping: {original_text}")
            else:
                final_result["Address"] = original_text  
                assigned_entries.add(original_text)
                print(f"📌 Assigned {original_text} to Address via [Regex/BERT] (Confidence: {confidences['Address'][1]:.2f})")

    #DEBUG: Show all address assignments before final result processing
    #print(f"\n✅ FINAL Assigned Address: {final_result.get('Address', '❌ No Address Assigned')}")

    # Step 2: Assign Unique Categories Based on Highest Confidence
    sorted_entries = sorted(results, key=lambda x: max(x[2].values(), key=lambda y: y[1])[1], reverse=True)

    for original_text, translated_text, confidences, eliminate_name in sorted_entries:
        if original_text in assigned_entries:  
            continue

        if eliminate_name:
            confidences["Name"] = ("Filtered", 0.0)

        # Find the best category after elimination
        best_category, (source, best_confidence) = max(
            confidences.items(), key=lambda x: x[1][1]
        )

        if best_category in UNIQUE_CATEGORIES and best_category not in unique_category_assignments:
            unique_category_assignments[best_category] = original_text
            assigned_entries.add(original_text)
            print(f"📌 Assigned {original_text} to {best_category} via [{source}] (Confidence: {best_confidence:.2f})")

    final_result.update(unique_category_assignments)

    # ------------------------ Fallback Logic for Missing Categories ------------------------
    for category in UNIQUE_CATEGORIES:
        if category not in final_result or not final_result[category]:
            remaining_candidates = [
                (original_text, confidences[category])
                for original_text, _, confidences, _ in sorted_entries
                if original_text not in assigned_entries and category in confidences and confidences[category][1] > 0.01
            ]

            if remaining_candidates:
                best_fallback, (fallback_source, fallback_confidence) = max(remaining_candidates, key=lambda x: x[1][1])
                final_result[category] = best_fallback
                print(f"📌 Assigned fallback {category}: {best_fallback} via [{fallback_source}] (Confidence: {fallback_confidence:.2f})")

    print("\n🔹 Final Structured Classification Result:")
    for key, value in final_result.items():
        print(f"📌 {key}: {value}")

    return final_result

def renormalize_confidences(all_entries, assigned_category):
    """Renormalizes confidence scores for all remaining entries after a category is assigned."""
    
    for entry in all_entries:
        confidences = entry[1]  

        if assigned_category in confidences:
            del confidences[assigned_category] 

        remaining_scores = [score for _, score in confidences.values()]
        total_score = sum(remaining_scores)

        if total_score > 0:  
            for category in confidences:
                source, score = confidences[category]
                confidences[category] = (source, score / total_score)
        else: 
            if len(confidences) == 1:
                category = list(confidences.keys())[0]
                confidences[category] = (confidences[category][0], 1.0)
    
    return all_entries 


# ------------------------ RUN CLASSIFIER WITH FULL TEXT INPUT ------------------------
if __name__ == "__main__":
    print("Multi-Layer Business Card Classifier Loaded")
    print("Enter multiple lines. Press 'Enter' on an empty line to process the input.\n")

    while True:
        lines = []
        print("\ninput business card information:")
        while True:
            line = input()
            if line.strip() == "":  # Stop collecting input when empty line is entered
                break
            lines.append(line)

        if not lines:
            print("\n🚀 Classification Completed. Goodbye!")
            break

        user_input = "\n".join(lines)
        classify_text(user_input)
