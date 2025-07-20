#!/bin/bash

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run Streamlit app
streamlit run main.py --server.port $PORT --server.enableCORS false

