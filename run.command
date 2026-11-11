#!/bin/bash
# 1-Click Launcher for JobShield
cd "$(dirname "$0")"
source .venv/bin/activate
PYTHONPATH=. streamlit run app.py
