# Multi-Layer Business Card Information Classifier

This project is a **multi-layer text classifier** designed to extract structured information from business card data. It combines:

1. **Regex-based rules** for phone numbers, emails, websites, addresses, and company suffixes.
2. **Named Entity Recognition (NER)** using spaCy to detect names, companies, and locations.
3. **Fine-tuned BERT model** (DistilBERT) for text classification with confidence scores.

The classifier handles multi-line input and attempts to classify each line into categories such as:

- Name
- Job Title
- Company
- Address
- Phone Number
- Email
- Website

extra requirements:
python -m spacy download en_core_web_sm

For the application to work properly please be sure the "MODEL_PATH" variable is set correctly.
This model is experimental and not %100 accurate, please check important data.
