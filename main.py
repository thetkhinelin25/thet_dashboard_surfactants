# app.py
import streamlit as st

st.set_page_config(page_title="Surfactant Optimization App", layout="wide")

st.title("Welcome to the Surfactant Optimization App")
st.markdown("""
Use the sidebar to navigate between different pages:
- **Overview** for summary
- **UMAP Exploration**
- **Performances Prediction**
- **Features Prediction**
""")
