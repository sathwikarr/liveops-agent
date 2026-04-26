"""Analyst — schema-agnostic data-science workbench layered on top of agent/.

Stages:
    1. ingest     — multi-format reader, type inference, dataset classification
    2. eda        — auto exploratory data analysis
    3. clean      — cleaning steps with an audit log
    4. analysis   — RFM, cohorts, basket, elasticity, anomalies
    5. predict    — churn, stockout, demand
    6. recommend  — insight-driven business recommendations
    7. nlq/whatif/report — NL query, simulator, docx/pdf reports
    8. calendar/join/competitor — action calendar, multi-CSV joins, web context
"""
