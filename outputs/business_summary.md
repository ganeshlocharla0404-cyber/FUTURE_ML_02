# Task 2 Business Summary

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
- Category model accuracy: 100.00%
- Priority model accuracy: 100.00%
- Evaluation includes accuracy, weighted precision, weighted recall, weighted F1-score, and confusion matrices.

Note: The dataset is anonymized and simulated with clear operational signals. A production version should be validated on real company tickets before deployment.

## Business Value
The system helps a support manager reduce manual triage time, route tickets to the right team, and identify urgent tickets earlier. This can improve first-response time, reduce backlog, and increase customer satisfaction.

## Sample Predictions
- Ticket: Payment was deducted twice and this has revenue impact for customers affected by billing errors. | Category: Billing | Priority: High
- Ticket: The dashboard export fails with a server error and our operations team cannot send reports to customers. | Category: Technical Issue | Priority: High
- Ticket: I need help understanding pricing plans for a growing startup team. | Category: General Query | Priority: Medium
- Ticket: A new employee cannot activate their invite link and needs account access today. | Category: Account | Priority: High
- Ticket: Webhook delivery is delayed and production order sync is blocked for many customers. | Category: Technical Issue | Priority: High
