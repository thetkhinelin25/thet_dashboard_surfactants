import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import zipfile
import io
from typing import Tuple, Dict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from umap_helpers.clean import clean_data, simplify_categories, transform_cmc
from sklearn.model_selection import KFold

# --- Config ---
st.set_page_config(layout="wide")
st.title("Page 3: Performance Prediction")
st.markdown("Use pre-trained models to predict performance or retrain with your own data.")

# --- Define Columns ---
feature_cols = [
    'primary surfactant name', 'secondary surfactant name', 'tertiary surfactant name',
    'primary surfactant level (%)', 'secondary surfactant level (%)', 'tertiary surfactant level (%)'
]
regression_targets = ['initial ph', 'appearance absorption value', 'height foam (mm)', 'mildness']
classification_targets = ['clarity', 'colour', 'physical state', 'cmc']

categorical_cols = ['primary surfactant name', 'secondary surfactant name', 'tertiary surfactant name']
numeric_cols = ['primary surfactant level (%)', 'secondary surfactant level (%)', 'tertiary surfactant level (%)']

# --- Session State Init ---
if "user_input" not in st.session_state:
    st.session_state.user_input = {}
if "new_samples" not in st.session_state:
    st.session_state.new_samples = []
if "custom_input" not in st.session_state:
    st.session_state.custom_input = {}
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}


# --- Reusable Training Function ---
@st.cache_data
def train_models(df: pd.DataFrame) -> Tuple:
    label_encoders: Dict[str, LabelEncoder] = {}

    # --- Label Encoding ---
    for col in categorical_cols + classification_targets:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # --- Standardize Inputs ---
    scaler_X = StandardScaler()
    df[numeric_cols] = scaler_X.fit_transform(df[numeric_cols])

    scaler_Y_dict = {target: StandardScaler() for target in regression_targets}
    for target in regression_targets:
        df[[target]] = scaler_Y_dict[target].fit_transform(df[[target]])

    # --- Feature & Target Splits ---
    X = df[feature_cols]
    Y_clf = df[classification_targets]

    # --- Initialize Regression Model Container ---
    trained_regressors = {}

    # 1. RF for 'height foam (mm)'
    reg = RandomForestRegressor(random_state=99)
    grid = GridSearchCV(reg, {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }, cv=KFold(n_splits=3, shuffle=True, random_state=9), n_jobs=-1, scoring='r2')
    grid.fit(X, df[['height foam (mm)']])
    trained_regressors['height foam (mm)'] = grid

    # 2. Use GBM for remaining: initial ph', 'mildness', 'appearance absorption value'
    gbm_targets = ['initial ph', 'mildness', 'appearance absorption value']
    reg = GradientBoostingRegressor(random_state=99)
    grid = GridSearchCV(reg, {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0]
    }, cv=KFold(n_splits=3, shuffle=True, random_state=9), n_jobs=-1, scoring='r2')
    multi_reg_model = MultiOutputRegressor(grid)
    multi_reg_model.fit(X, df[gbm_targets])
    for i, target in enumerate(gbm_targets):
        trained_regressors[target] = multi_reg_model.estimators_[i]

    # -- Classification model --
    # Use GBM for 'clarity', 'colour', 'physical state', 'cmc'
    classifier = GradientBoostingClassifier(random_state=99)
    clf_param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.05],
        'max_depth': [7, 8],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 3],
        'subsample': [1.0]
    }
    clf_grid = GridSearchCV(classifier, clf_param_grid,
                            cv=KFold(n_splits=3, shuffle=True, random_state=9),
                            n_jobs=-1, scoring='balanced_accuracy')
    multi_clf_model = MultiOutputClassifier(clf_grid)
    multi_clf_model.fit(X, Y_clf)

    # --- Save Everything ---
    os.makedirs("user_temp_models", exist_ok=True)
    joblib.dump(trained_regressors, "user_temp_models/trained_regressors.pkl")
    joblib.dump(multi_clf_model, "user_temp_models/multi_clf_model.pkl")
    joblib.dump(scaler_X, "user_temp_models/scaler_X.pkl")
    joblib.dump(scaler_Y_dict, "user_temp_models/scaler_Y_dict.pkl")
    joblib.dump(label_encoders, "user_temp_models/label_encoders.pkl")

    return trained_regressors, multi_clf_model, scaler_X, scaler_Y_dict, label_encoders




# --- PART 1: Train Model ---
with st.expander("Train Model", expanded=False):
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"], key="retrain_file")

    for key in ["reg_model", "clf_model", "sx", "sy", "le"]:
        st.session_state.setdefault(key, None)

    if uploaded_file and st.button("Train New Model"):
        df = clean_data(pd.read_excel(uploaded_file))
        df = simplify_categories(df)
        df['cmc'] = df['cmc'].apply(transform_cmc)
        df = df[feature_cols + regression_targets + classification_targets]
        try:
            reg_model, clf_model, sx, sy, le = train_models(df)

            st.session_state['reg_model'] = reg_model
            st.session_state['clf_model'] = clf_model
            st.session_state['sx'] = sx
            st.session_state['sy'] = sy
            st.session_state['le'] = le

            st.success("✅ Training complete. You can now download the trained models.")

        except Exception as e:
            st.error(f"❌ Training failed: {e}")

    if all(k in st.session_state and st.session_state[k] is not None for k in ['reg_model', 'clf_model', 'sx', 'sy', 'le']):
        os.makedirs("new_trained_models", exist_ok=True)
        model_files = {
            "reg": st.session_state['reg_model'],
            "clf": st.session_state['clf_model'],
            "sx": st.session_state['sx'],
            "sy": st.session_state['sy'],
            "le": st.session_state['le']
        }
        for name, obj in model_files.items():
            joblib.dump(obj, f"new_trained_models/{name}.pkl")

        zip_path = "new_trained_models_bundle.zip"
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for name in model_files.keys():
                zipf.write(f"new_trained_models/{name}.pkl", arcname=f"{name}.pkl")

        with open(zip_path, "rb") as f:
            st.download_button("Download Trained Model Files (ZIP)", f, file_name="model_bundle.zip")


# --- PART 2: Make Prediction ---
with st.expander("Make Prediction", expanded=False):
    keys = ["reg_model", "clf_model", "sx", "sy", "le"]
    if all(k in st.session_state and st.session_state[k] is not None for k in keys):
        reg_model = st.session_state["reg_model"]
        clf_model = st.session_state["clf_model"]
        sx = st.session_state["sx"]
        sy = st.session_state["sy"]
        le = st.session_state["le"]

        st.markdown("### Predict Using Trained Model")

        # --- Initialize session state ---
        for col in categorical_cols:
            if f"r_{col}_" not in st.session_state:
                st.session_state[f"r_{col}_"] = le[col].classes_[0]

        for col in numeric_cols:
            if f"r_{col}_" not in st.session_state:
                st.session_state[f"r_{col}_"] = 0.0

        # --- Callback functions ---
        def update_categorical(col):
            st.session_state[f"r_{col}_"] = st.session_state[f"widget_{col}"]

        def update_numeric(col):
            st.session_state[f"r_{col}_"] = st.session_state[f"widget_{col}"]

        # --- UI for Categorical Inputs ---
        user_input = {}
        cols1 = st.columns(3)
        for i, col in enumerate(categorical_cols):
            with cols1[i]:
                st.selectbox(
                    col,
                    le[col].classes_,
                    index=list(le[col].classes_).index(st.session_state[f"r_{col}_"]),
                    key=f"widget_{col}",
                    on_change=update_categorical,
                    args=(col,)
                )
                user_input[col] = st.session_state[f"widget_{col}"]

        # --- UI for Numeric Inputs ---
        cols2 = st.columns(3)
        for i, col in enumerate(numeric_cols):
            with cols2[i]:
                st.number_input(
                    col,
                    value=st.session_state[f"r_{col}_"],
                    format="%.2f",
                    key=f"widget_{col}",
                    on_change=update_numeric,
                    args=(col,)
                )
                user_input[col] = st.session_state[f"widget_{col}"]

        if st.button("Predict_1"):
            input_df = pd.DataFrame([user_input])
            for col in categorical_cols:
                input_df[col] = le[col].transform(input_df[col])
            input_df[numeric_cols] = sx.transform(input_df[numeric_cols])

            reg_preds = []
            for col in regression_targets:
                model = reg_model[col]
                pred = model.predict(input_df)
                if len(pred.shape) == 1:
                    pred = pred.reshape(-1, 1)
                inv_pred = sy[col].inverse_transform(pred)
                reg_preds.append(inv_pred[0][0])

            clf_preds = clf_model.predict(input_df)
            decoded = {
                col: le[col].inverse_transform([clf_preds[0][i]])[0]
                for i, col in enumerate(classification_targets)
            }

            st.success("New Model Prediction Results")
            for i, col in enumerate(regression_targets):
                st.write(f"**{col}**: {reg_preds[i]:.2f}")
            for col in classification_targets:
                st.write(f"**{col}**: {decoded[col]}")
    else:
        st.warning("⚠️ Please train the model first.")


# --- PART 3: Upload Model and Make Prediction ---
with st.expander("Upload Model and Make Prediction", expanded=False):
    st.markdown("### Reupload Trained Files")
    cols = st.columns(5)
    labels = ["reg.pkl", "clf.pkl", "sx.pkl", "sy.pkl", "le.pkl"]
    keys = ['reg', 'clf', 'sx', 'sy', 'le']

    for col, key, label in zip(cols, keys, labels):
        with col:
            uploaded = st.file_uploader(label, type=["pkl"], key=f"upload_{key}")
            if uploaded is not None:
                st.session_state.uploaded_files[key] = uploaded.read()
                st.markdown(f"✅ **{label} uploaded**")
            elif key in st.session_state.uploaded_files:
                st.markdown(f"✅ **{label} uploaded**")
            else:
                st.markdown(f"❌ **{label} not uploaded**")

    if all(k in st.session_state.uploaded_files for k in keys):
        reg_model = joblib.load(io.BytesIO(st.session_state.uploaded_files["reg"]))
        clf_model = joblib.load(io.BytesIO(st.session_state.uploaded_files["clf"]))
        sx = joblib.load(io.BytesIO(st.session_state.uploaded_files["sx"]))
        sy = joblib.load(io.BytesIO(st.session_state.uploaded_files["sy"]))
        le = joblib.load(io.BytesIO(st.session_state.uploaded_files["le"]))

        st.markdown("---")
        st.markdown("### Predict Using New Model")

        # --- Initialize session state ---
        for col in categorical_cols:
            if f"r_{col}2_" not in st.session_state:
                st.session_state[f"r_{col}2_"] = le[col].classes_[0]

        for col in numeric_cols:
            if f"r_{col}2_" not in st.session_state:
                st.session_state[f"r_{col}2_"] = 0.0

        # --- Callback functions ---
        def update_categorical(col):
            st.session_state[f"r_{col}2_"] = st.session_state[f"widget2_{col}"]

        def update_numeric(col):
            st.session_state[f"r_{col}2_"] = st.session_state[f"widget2_{col}"]

        # --- UI for Categorical Inputs ---
        user_input = {}
        cols1 = st.columns(3)
        for i, col in enumerate(categorical_cols):
            with cols1[i]:
                st.selectbox(
                    col,
                    le[col].classes_,
                    index=list(le[col].classes_).index(st.session_state[f"r_{col}2_"]),
                    key=f"widget2_{col}",
                    on_change=update_categorical,
                    args=(col,)
                )
                user_input[col] = st.session_state[f"widget2_{col}"]

        # --- UI for Numeric Inputs ---
        cols2 = st.columns(3)
        for i, col in enumerate(numeric_cols):
            with cols2[i]:
                st.number_input(
                    col,
                    value=st.session_state[f"r_{col}2_"],
                    format="%.2f",
                    key=f"widget2_{col}",
                    on_change=update_numeric,
                    args=(col,)
                )
                user_input[col] = st.session_state[f"widget2_{col}"]

        
        if st.button("Predict_2"):
            input_df = pd.DataFrame([user_input])
            for col in categorical_cols:
                input_df[col] = le[col].transform(input_df[col])
            input_df[numeric_cols] = sx.transform(input_df[numeric_cols])

            reg_preds = []
            for col in regression_targets:
                model = reg_model[col]
                pred = model.predict(input_df)
                if len(pred.shape) == 1:
                    pred = pred.reshape(-1, 1)
                inv_pred = sy[col].inverse_transform(pred)
                reg_preds.append(inv_pred[0][0])

            clf_preds = clf_model.predict(input_df)
            decoded = {
                col: le[col].inverse_transform([clf_preds[0][i]])[0]
                for i, col in enumerate(classification_targets)
            }

            st.success("New Model Prediction Results")
            for i, col in enumerate(regression_targets):
                st.write(f"**{col}**: {reg_preds[i]:.2f}")
            for col in classification_targets:
                st.write(f"**{col}**: {decoded[col]}")