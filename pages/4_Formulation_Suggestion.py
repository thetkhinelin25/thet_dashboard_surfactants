import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern, Kernel
from scipy.optimize import differential_evolution, NonlinearConstraint
from collections import OrderedDict

st.set_page_config(page_title="BRO Dashboard", layout="wide")
st.title("🧪 Bayesian Reaction Optimization (BRO) Dashboard")

clarity_classes = ['clear', 'slightly turbid', 'turbid']
colour_classes = ['colourless', 'grey', 'white', 'yellow']
state_classes = ['liquid', 'non-liquid']
cmc_classes = ['greater than 0.03', 'less than or equal 0.03']

cat_inputs = ["primary surfactant name", "secondary surfactant name", "tertiary surfactant name"]
cont_inputs = ["primary surfactant level (%)", "secondary surfactant level (%)", "tertiary surfactant level (%)"]
categorical_targets = ["clarity", "colour", "physical state", "cmc"]
regression_targets = ["initial ph", "appearance absorption value", "height foam (mm)", "mildness"]

clarity_map = {'clear': 0, 'slightly turbid': 1, 'turbid': 2}
colour_map = {'colourless': 0, 'grey': 1, 'white': 2, 'yellow': 3}
state_map = {'liquid': 0, 'non-liquid': 1}
cmc_map = {'greater than 0.03': 0, 'less than or equal 0.03': 1}


with st.expander("Train GP Models", expanded=False):
    # --- Upload Section ---
    uploaded_file = st.file_uploader("Upload Excel Dataset", type=["xlsx"])

    if uploaded_file:
        df = pd.read_excel(uploaded_file, sheet_name="Data_Summary_PST")
        df.columns = df.columns.str.strip().str.lower()
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
        df['cmc'] = df['cmc'].apply(lambda x: 'greater than 0.03' if float(x) > 0.03 else 'less than or equal 0.03')

        # Mappings for original class names
        df['clarity'] = df['clarity'].map(clarity_map)
        df['colour'] = df['colour'].map(colour_map)
        df['physical state'] = df['physical state'].map(state_map)
        df['cmc'] = df['cmc'].map(cmc_map)

        class_name_maps = {
            'clarity': {v: k for k, v in clarity_map.items()},
            'colour': {v: k for k, v in colour_map.items()},
            'physical state': {v: k for k, v in state_map.items()},
            'cmc': {v: k for k, v in cmc_map.items()}
        }

        if st.button("Train GP Models"):
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            cat_encoded = encoder.fit_transform(df[cat_inputs])
            cat_group_sizes = [len(c) for c in encoder.categories_]
            X = np.concatenate([cat_encoded, df[cont_inputs].values], axis=1)

            class TransformedKernel(Kernel):
                def __init__(self, base_kernel, cat_group_sizes):
                    self.base_kernel = base_kernel
                    self.cat_group_sizes = cat_group_sizes
                def _transform(self, X):
                    X = X.copy()
                    start = 0
                    for size in self.cat_group_sizes:
                        end = start + size
                        max_idx = np.argmax(X[:, start:end], axis=1)
                        X[:, start:end] = 0
                        X[np.arange(X.shape[0]), start + max_idx] = 1
                        start = end
                    return X
                def __call__(self, X, Y=None, eval_gradient=False):
                    return self.base_kernel(self._transform(X), self._transform(Y) if Y is not None else None, eval_gradient)
                def diag(self, X): return self.base_kernel.diag(self._transform(X))
                def is_stationary(self): return self.base_kernel.is_stationary()

            kernel = TransformedKernel(Matern(nu=2.5), cat_group_sizes)

            scalers, y_scaled_dict = {}, {}
            for col in regression_targets:
                scaler = MinMaxScaler()
                y_scaled = scaler.fit_transform(df[[col]])
                scalers[col], y_scaled_dict[col] = scaler, y_scaled

            gp_classifiers = {col: GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=5).fit(X, df[col]) for col in categorical_targets}
            gp_regressors = {col: GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True).fit(X, y_scaled_dict[col].ravel()) for col in regression_targets}

            def split_x(x): return x[:cat_encoded.shape[1]], x[cat_encoded.shape[1]:]

            def apply_logical_constraints(x):
                x = np.asarray(x).flatten()
                cat_part, cont_part = split_x(x)
                corrected_cat = cat_part.copy()
                start = 0
                for size in cat_group_sizes:
                    end = start + size
                    max_idx = np.argmax(corrected_cat[start:end])
                    corrected_cat[start:end] = 0
                    corrected_cat[start + max_idx] = 1
                    start = end
                decoded = encoder.inverse_transform([corrected_cat])[0]
                p_level = cont_part[0] if decoded[0] != 'no' else 0.0
                s_level = cont_part[1] if decoded[1] != 'no' else 0.0
                t_level = cont_part[2] if decoded[2] != 'no' else 0.0
                return np.concatenate([corrected_cat, [p_level, s_level, t_level]]).reshape(1, -1)

            def sum_to_15(x):
                x_corrected = apply_logical_constraints(x)
                return float(np.sum(x_corrected[0, -3:]))
            
            # ✅ Register the function in session state
            st.session_state["sum_to_15"] = sum_to_15

            st.session_state.update({
                "encoder": encoder,
                "gp_classifiers": gp_classifiers,
                "gp_regressors": gp_regressors,
                "scalers": scalers,
                "apply_constraints": apply_logical_constraints,
                "split_x": split_x,
                "cat_encoded": cat_encoded,
                "cat_group_sizes": cat_group_sizes,
                "class_name_maps": class_name_maps
            })
            st.success("✅ GP Models Trained.")

# --- BRO with Specified Targets ---
with st.expander("🎯 BRO: Specified Targets"):
    if "gp_regressors" in st.session_state:
        target_class = {}
        target_scaled = {}
        target_weights = {}

        st.markdown("### 🧮 Numeric Targets")
        for col in regression_targets:
            col1, col2 = st.columns([2, 1])
            with col1:
                val = st.number_input(f"Target value for {col}", value=0.0, key=f"val_{col}")
            with col2:
                weight = st.selectbox("Importance", ["Not Important", "Normal", "Important", "Very Important"], key=f"w_{col}")
            target_scaled[col] = st.session_state["scalers"][col].transform([[val]])[0, 0]
            target_weights[col] = ["Not Important", "Normal", "Important", "Very Important"].index(weight)

        st.markdown("### 🧠 Categorical Targets")
        for col in categorical_targets:
            name_map = st.session_state["class_name_maps"][col]
            options = list(name_map.values())
            inv_map = {v: k for k, v in name_map.items()}
            col1, col2 = st.columns([2, 1])
            with col1:
                selected = st.selectbox(f"{col.title()} class", options, key=f"class_{col}")
            with col2:
                weight = st.selectbox("Importance", ["Not Important", "Normal", "Important", "Very Important"], key=f"w_cat_{col}")
            val = inv_map[selected]
            target_class[col] = val
            target_weights[col] = ["Not Important", "Normal", "Important", "Very Important"].index(weight)

        def acquisition_specified(x):
            x_corrected = st.session_state["apply_constraints"](x)
            score = 0.0
            for col in categorical_targets:
                model = st.session_state["gp_classifiers"][col]
                proba = model.predict_proba(x_corrected)[0]
                score += -target_weights[col] * proba[target_class[col]]
            for col in regression_targets:
                model = st.session_state["gp_regressors"][col]
                mu, _ = model.predict(x_corrected, return_std=True)
                diff = (mu[0] - target_scaled[col]) ** 2
                score += target_weights[col] * diff
            return float(score)

        if st.button("Generate Suggestions (Specified Targets)"):
            with st.spinner("🔄 Optimizing formulation suggestions..."):
                bounds = [(0, 1)] * st.session_state["cat_encoded"].shape[1] + [(0, 15)] * 3
                constraint = NonlinearConstraint(st.session_state["sum_to_15"], 0.01, 15)
                seeds = range(9, 200, 10)
                suggestions = []
                for seed in seeds:
                    result = differential_evolution(acquisition_specified, bounds, constraints=(constraint,),
                                                    seed=seed, maxiter=10, popsize=10, polish=False, workers=1)
                    x_final = st.session_state["apply_constraints"](result.x).flatten()
                    score = acquisition_specified(result.x)
                    suggestions.append((score, x_final))

                unique = OrderedDict()
                for score, x in sorted(suggestions, key=lambda x: x[0]):
                    cat_part, cont_part = st.session_state["split_x"](x)
                    decoded = tuple(st.session_state["encoder"].inverse_transform([cat_part])[0])
                    levels = tuple(np.round(cont_part, 2))
                    key = decoded + levels
                    if key not in unique:
                        unique[key] = x
                    if len(unique) >= 5:
                        break

                st.markdown("#### 🔝 Top 5 Suggested Formulations (Specified Targets):")
                for i, x in enumerate(unique.values(), 1):
                    cat_part, cont_part = st.session_state["split_x"](x)
                    decoded = st.session_state["encoder"].inverse_transform([cat_part])[0]
                    p, s, t = cont_part
                    st.write(f"**Formulation #{i}:**")
                    st.write(f"- Primary: {decoded[0]} | {p:.2f}%")
                    st.write(f"- Secondary: {decoded[1]} | {s:.2f}%")
                    st.write(f"- Tertiary: {decoded[2]} | {t:.2f}%")
    
    else:
        st.warning("⚠️ Please train the GP models first.")


# --- BRO with Max/Min/Close-To ---
with st.expander("⚙️ BRO: Maximize / Minimize / Close-To"):
    if "gp_regressors" in st.session_state:
        target_objective = {}
        target_reference = {}
        target_weights = {}
        target_class_soft_weights = {}

        st.markdown("### 🧮 Numeric Targets")
        for col in regression_targets:
            col1, col2 = st.columns([2, 1])
            with col1:
                obj = st.selectbox(f"{col.title()} Objective", ["maximize", "minimize", "close_to"], key=f"obj_{col}")
            with col2:
                imp = st.selectbox("Importance", ["Not Important", "Normal", "Important", "Very Important"], key=f"imp_{col}")
            target_objective[col] = obj
            target_weights[col] = ["Not Important", "Normal", "Important", "Very Important"].index(imp)

            if obj == "close_to":
                ref = st.number_input(f"Reference value for {col}", value=0.0, key=f"ref_{col}")
                scaled_ref = st.session_state["scalers"][col].transform([[ref]])[0, 0]
                target_reference[col] = scaled_ref

        st.markdown("### 🧠 Categorical Targets")
        for col in categorical_targets:
            with st.container():
                st.markdown(f"**🎯 {col.title()}**")
                class_weights = {}
                class_map = st.session_state["class_name_maps"][col]
                for label_id, label_name in class_map.items():
                    w = st.slider(
                        f"{label_name}", min_value=0.0, max_value=1.0, step=0.05,
                        key=f"{col}_{label_name}_soft"
                    )
                    class_weights[label_id] = w
                target_class_soft_weights[col] = class_weights

                var_imp = st.selectbox("Variable Importance", ["Not Important", "Normal", "Important", "Very Important"], key=f"imp_{col}")
                target_weights[col] = ["Not Important", "Normal", "Important", "Very Important"].index(var_imp)

        def acquisition_flexible(x):
            x_corrected = st.session_state["apply_constraints"](x)
            score = 0.0

            for col in categorical_targets:
                proba = st.session_state["gp_classifiers"][col].predict_proba(x_corrected)[0]
                soft_map = target_class_soft_weights[col]
                weighted_prob = sum(soft_map.get(i, 0) * proba[i] for i in range(len(proba)))
                score += -target_weights[col] * weighted_prob

            for col in regression_targets:
                mu, _ = st.session_state["gp_regressors"][col].predict(x_corrected, return_std=True)
                mean_val = mu[0]
                if target_objective[col] == "maximize":
                    score += -target_weights[col] * mean_val
                elif target_objective[col] == "minimize":
                    score += target_weights[col] * mean_val
                elif target_objective[col] == "close_to":
                    target_val = target_reference[col]
                    score += target_weights[col] * (mean_val - target_val) ** 2

            return float(score)

        if st.button("Generate Suggestions (Max/Min/Close-To)"):
            with st.spinner("🔄 Optimizing formulation suggestions..."):
                bounds = [(0, 1)] * st.session_state["cat_encoded"].shape[1] + [(0, 15)] * 3
                constraint = NonlinearConstraint(st.session_state["sum_to_15"], 0.01, 15)
                seeds = range(9, 200, 10)
                suggestions = []

                for seed in seeds:
                    result = differential_evolution(
                        acquisition_flexible, bounds,
                        constraints=(constraint,),
                        seed=seed, maxiter=10, popsize=10,
                        polish=False, workers=1
                    )
                    x_final = st.session_state["apply_constraints"](result.x).flatten()
                    score = acquisition_flexible(result.x)
                    suggestions.append((score, x_final))

                unique = OrderedDict()
                for score, x in sorted(suggestions, key=lambda x: x[0]):
                    cat_part, cont_part = st.session_state["split_x"](x)
                    decoded = tuple(st.session_state["encoder"].inverse_transform([cat_part])[0])
                    levels = tuple(np.round(cont_part, 2))
                    key = decoded + levels
                    if key not in unique:
                        unique[key] = x
                    if len(unique) >= 5:
                        break

                st.markdown("#### 🔝 Top 5 Suggested Formulations (Max/Min/Close-To):")
                for i, x in enumerate(unique.values(), 1):
                    cat_part, cont_part = st.session_state["split_x"](x)
                    decoded = st.session_state["encoder"].inverse_transform([cat_part])[0]
                    p, s, t = cont_part
                    st.write(f"**Formulation #{i}:**")
                    st.write(f"- Primary: {decoded[0]} | {p:.2f}%")
                    st.write(f"- Secondary: {decoded[1]} | {s:.2f}%")
                    st.write(f"- Tertiary: {decoded[2]} | {t:.2f}%")
    
    else:
        st.warning("⚠️ Please train the GP models first.")



