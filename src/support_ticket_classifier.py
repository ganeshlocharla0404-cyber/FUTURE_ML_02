from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RANDOM_SEED = 42


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "i",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "our",
    "please",
    "that",
    "the",
    "this",
    "to",
    "was",
    "we",
    "with",
    "you",
}


CATEGORY_TEMPLATES = {
    "Billing": [
        "Invoice amount is incorrect after plan renewal",
        "Payment failed but money was deducted from bank account",
        "Need refund for duplicate subscription charge",
        "Tax details are missing from the latest invoice",
        "Card charged twice for the same billing cycle",
        "Unable to download receipt for annual plan",
    ],
    "Technical Issue": [
        "Application crashes when uploading a large csv file",
        "Dashboard is loading slowly after the latest update",
        "API returns timeout error during order sync",
        "Mobile app freezes on the login screen",
        "Report export fails with server error",
        "Integration webhook is not sending payloads",
    ],
    "Account": [
        "Unable to reset password using registered email",
        "Need to change account owner for the workspace",
        "User invite link expired before activation",
        "Two factor authentication code is not working",
        "Please remove inactive users from our account",
        "Cannot update profile and organization details",
    ],
    "General Query": [
        "Need information about available pricing plans",
        "Can you explain how the trial period works",
        "Where can I find documentation for reports",
        "Do you support custom onboarding for teams",
        "Question about product features and limits",
        "Need guidance on choosing the right package",
    ],
}


PRIORITY_HINTS = {
    "High": [
        "urgent",
        "production blocked",
        "business critical",
        "customers affected",
        "cannot access",
        "revenue impact",
        "security concern",
    ],
    "Medium": [
        "important",
        "affecting team",
        "needs attention",
        "before tomorrow",
        "workflow delayed",
        "partial outage",
    ],
    "Low": [
        "when possible",
        "minor question",
        "not urgent",
        "future reference",
        "small clarification",
        "nice to have",
    ],
}


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    words = [word for word in text.split() if word not in STOPWORDS and len(word) > 1]
    return " ".join(words)


def generate_support_tickets(output_path: Path, rows_per_category: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    channels = ["email", "web form", "chat", "mobile app", "help desk"]
    customer_types = ["startup", "enterprise", "agency", "retail business", "education team"]
    closing_phrases = [
        "Can your team help us resolve this?",
        "We need a clear update from support.",
        "Please share the next steps.",
        "This is affecting our daily workflow.",
        "Let us know if you need more information.",
    ]

    rows = []
    ticket_id = 1001
    for category, templates in CATEGORY_TEMPLATES.items():
        for _ in range(rows_per_category):
            priority = rng.choice(["High", "Medium", "Low"], p=[0.24, 0.46, 0.30])
            template = rng.choice(templates)
            hint = rng.choice(PRIORITY_HINTS[priority])
            channel = rng.choice(channels)
            customer_type = rng.choice(customer_types)
            affected_users = int(rng.choice([1, 2, 3, 5, 10, 25, 80], p=[0.28, 0.2, 0.16, 0.14, 0.11, 0.07, 0.04]))
            sentiment = rng.choice(["frustrated", "concerned", "confused", "blocked", "waiting"])
            text = (
                f"{template}. The request came through {channel} from a {customer_type}. "
                f"The customer is {sentiment} because {hint}. "
                f"Affected users: {affected_users}. {rng.choice(closing_phrases)}"
            )

            rows.append(
                {
                    "ticket_id": f"TCK-{ticket_id}",
                    "ticket_text": text,
                    "category": category,
                    "priority": priority,
                    "channel": channel,
                    "affected_users": affected_users,
                }
            )
            ticket_id += 1

    df = pd.DataFrame(rows).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    df["clean_text"] = df["ticket_text"].map(clean_text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def load_or_create_data() -> pd.DataFrame:
    data_path = DATA_DIR / "support_tickets.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
        df["clean_text"] = df["ticket_text"].map(clean_text)
        return df
    return generate_support_tickets(data_path)


def build_category_model() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.92,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", LinearSVC(random_state=RANDOM_SEED)),
        ]
    )


def build_priority_model() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.92,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1200,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def evaluate_model(model: Pipeline, x_test: pd.Series, y_test: pd.Series, labels: list[str], output_name: str) -> dict:
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        labels=labels,
        average="weighted",
        zero_division=0,
    )
    report = classification_report(y_test, predictions, labels=labels, output_dict=True, zero_division=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, labels=labels, cmap="Blues", ax=ax)
    ax.set_title(output_name.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{output_name}.png", dpi=160)
    plt.close(fig)

    return {
        "accuracy": float(accuracy),
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "weighted_f1": float(f1),
        "classification_report": report,
    }


def predict_examples(category_model: Pipeline, priority_model: Pipeline) -> pd.DataFrame:
    examples = [
        "Payment was deducted twice and this has revenue impact for customers affected by billing errors.",
        "The dashboard export fails with a server error and our operations team cannot send reports to customers.",
        "I need help understanding pricing plans for a growing startup team.",
        "A new employee cannot activate their invite link and needs account access today.",
        "Webhook delivery is delayed and production order sync is blocked for many customers.",
    ]
    cleaned_examples = [clean_text(text) for text in examples]
    return pd.DataFrame(
        {
            "ticket_text": examples,
            "clean_text": cleaned_examples,
            "predicted_category": category_model.predict(cleaned_examples),
            "predicted_priority": priority_model.predict(cleaned_examples),
        }
    )


def write_business_summary(metrics: dict, examples: pd.DataFrame) -> None:
    category_accuracy = metrics["category_model"]["accuracy"]
    priority_accuracy = metrics["priority_model"]["accuracy"]
    example_lines = []
    for _, row in examples.iterrows():
        example_lines.append(
            f"- Ticket: {row['ticket_text']} | Category: {row['predicted_category']} | "
            f"Priority: {row['predicted_priority']}"
        )
    example_text = "\n".join(example_lines)

    summary = f"""# Task 2 Business Summary

## System Overview
This project classifies incoming support tickets into business categories and predicts a priority level so support teams can route work faster.

## Categories
- Billing
- Technical Issue
- Account
- General Query

## Priority Levels
- High: urgent, blocked, revenue-impacting, customer-impacting, or security-sensitive issues
- Medium: important workflow issues that need timely support
- Low: general questions, minor requests, and non-urgent clarifications

## Model Performance
- Category model accuracy: {category_accuracy:.2%}
- Priority model accuracy: {priority_accuracy:.2%}
- Evaluation includes accuracy, weighted precision, weighted recall, weighted F1-score, and confusion matrices.

Note: The dataset is anonymized and simulated with clear operational signals. A production version should be validated on real company tickets before deployment.

## Business Value
The system helps a support manager reduce manual triage time, route tickets to the right team, and identify urgent tickets earlier. This can improve first-response time, reduce backlog, and increase customer satisfaction.

## Sample Predictions
{example_text}
"""
    (OUTPUT_DIR / "business_summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train support ticket category and priority classifiers.")
    parser.add_argument("--skip-generate", action="store_true", help="Use the existing CSV if present.")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_generate:
        generate_support_tickets(DATA_DIR / "support_tickets.csv")

    df = load_or_create_data()
    train_df, test_df = train_test_split(
        df,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=df["category"],
    )
    x_train = train_df["clean_text"]
    x_test = test_df["clean_text"]
    y_category_train = train_df["category"]
    y_category_test = test_df["category"]
    y_priority_train = train_df["priority"]
    y_priority_test = test_df["priority"]

    category_model = build_category_model()
    priority_model = build_priority_model()
    category_model.fit(x_train, y_category_train)
    priority_model.fit(x_train, y_priority_train)

    metrics = {
        "category_model": evaluate_model(
            category_model,
            x_test,
            y_category_test,
            sorted(df["category"].unique()),
            "category_confusion_matrix",
        ),
        "priority_model": evaluate_model(
            priority_model,
            x_test,
            y_priority_test,
            ["High", "Medium", "Low"],
            "priority_confusion_matrix",
        ),
    }

    examples = predict_examples(category_model, priority_model)
    examples.to_csv(OUTPUT_DIR / "sample_predictions.csv", index=False)
    (OUTPUT_DIR / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_business_summary(metrics, examples)

    print("Category model")
    print(f"Accuracy: {metrics['category_model']['accuracy']:.4f}")
    print(f"Weighted F1: {metrics['category_model']['weighted_f1']:.4f}")
    print("Priority model")
    print(f"Accuracy: {metrics['priority_model']['accuracy']:.4f}")
    print(f"Weighted F1: {metrics['priority_model']['weighted_f1']:.4f}")
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
