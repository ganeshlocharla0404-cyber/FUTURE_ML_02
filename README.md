# FUTURE_ML_02 - Support Ticket Classification and Prioritization

Machine Learning Task 2 for the Future Interns ML internship.

## Project Overview

This project builds an NLP-based support ticket classification system. It reads customer support ticket text, predicts the ticket category, and assigns a priority level so support teams can route and respond to issues faster.

## Business Problem

Support teams often receive many tickets through email, chat, help desk portals, and web forms. Manual sorting can delay urgent issues and increase backlog. This project uses machine learning to support faster triage.

The system predicts:

- Ticket category: Billing, Technical Issue, Account, General Query
- Priority level: High, Medium, Low

## Dataset

The project uses a realistic anonymized support ticket dataset generated inside the pipeline. It includes:

- ticket text
- category label
- priority label
- support channel
- affected user count

The dataset simulates real operational support scenarios such as payment failures, account access problems, technical errors, and pricing questions.

Because the generated dataset contains clear operational signals for each class, evaluation scores can be very strong. In a production setting, the same pipeline should be validated on real company tickets before deployment.

## Methodology

1. Generated anonymized support ticket records.
2. Cleaned ticket text using lowercasing, punctuation handling, and stopword removal.
3. Converted text into TF-IDF features using unigrams and bigrams.
4. Trained a category classification model using Linear SVC.
5. Trained a priority classification model using Logistic Regression.
6. Evaluated both models using accuracy, weighted precision, weighted recall, weighted F1-score, and confusion matrices.
7. Created sample predictions and a business-readable summary.

## Project Structure

```text
FUTURE_ML_02/
  data/
    support_tickets.csv
  outputs/
    business_summary.md
    category_confusion_matrix.png
    model_metrics.json
    priority_confusion_matrix.png
    sample_predictions.csv
  src/
    support_ticket_classifier.py
  README.md
  requirements.txt
```

## How to Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/support_ticket_classifier.py
```

If your Python environment already has the libraries installed, run:

```bash
python src/support_ticket_classifier.py
```

## Deliverables

- Classification model: `src/support_ticket_classifier.py`
- Dataset: `data/support_tickets.csv`
- Model metrics: `outputs/model_metrics.json`
- Business summary: `outputs/business_summary.md`
- Sample predictions: `outputs/sample_predictions.csv`
- Visuals:
  - `outputs/category_confusion_matrix.png`
  - `outputs/priority_confusion_matrix.png`

## Business Interpretation

The model can help support teams automatically route tickets to the correct department and flag urgent tickets earlier. This improves response time, reduces manual triage effort, and helps managers understand ticket patterns.

## Skills Demonstrated

- NLP text preprocessing
- TF-IDF feature extraction
- Multi-class text classification
- Priority prediction
- Model evaluation with confusion matrices and classification metrics
- Business-focused ML explanation
