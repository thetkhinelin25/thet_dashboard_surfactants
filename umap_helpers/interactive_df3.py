import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.manifold import trustworthiness
import umap

def get_interactive_df3_original(df):
    # Clean column names: strip and lowercase
    df.columns = df.columns.str.strip().str.lower()

    # Clean all categorical values: strip and lowercase
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()

    # --- Selected columns ---
    selected_columns_features = ["primary surfactant name", "secondary surfactant name", "tertiary surfactant name",
                                "primary surfactant level (%)", "secondary surfactant level (%)", "tertiary surfactant level (%)"]
    
    selected_columns_performances = ["clarity", "colour", "physical state", "initial ph", 
                                    "appearance absorption value", "height foam (mm)", 
                                    "cmc", "mildness"]
    
    df_features_drop = df[selected_columns_features]

    # --- Drop performance-related columns ---
    df_performance_drop = df[selected_columns_performances]

    # --- Combine back ---
    df_combined = pd.concat([df_features_drop, df_performance_drop], axis=1)

    # Encode numeric and categorical features
    num_cols = df_combined.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df_combined.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(), cat_cols)
    ])

    df_encoded = preprocessor.fit_transform(df_combined)

    # --- UMAP ---
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    df_umap = umap_model.fit_transform(df_encoded)

    trust = trustworthiness(df_encoded, df_umap, n_neighbors=15)
    print(f"[UMAP3] Trustworthiness Score: {trust:.3f}")

    # --- Final Assembly ---
    df_plot = pd.DataFrame(df_umap, columns=["UMAP1", "UMAP2"])
    df_plot["Type"] = "Original"

    interactive_df3 = pd.concat([
        df_plot.reset_index(drop=True),
        df.reset_index(drop=True)
    ], axis=1)

    interactive_df3["Index"] = interactive_df3.index
    return interactive_df3


def get_interactive_df3_synthetic(original_data, synthetic_data):
    # Clean up string columns
    for df in [original_data, synthetic_data]:
        # Clean column names: strip and lowercase
        df.columns = df.columns.str.strip().str.lower()

        # Clean all categorical values: strip and lowercase
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            df[col] = df[col].astype(str).str.strip().str.lower()

    # --- Selected columns ---
    selected_columns_features = ["primary surfactant name", "secondary surfactant name", "tertiary surfactant name",
                                "primary surfactant level (%)", "secondary surfactant level (%)", "tertiary surfactant level (%)"]
    
    selected_columns_performances = ["clarity", "colour", "physical state", "initial ph", 
                                    "appearance absorption value", "height foam (mm)", 
                                    "cmc", "mildness"]
    
    original_data_features_drop = original_data[selected_columns_features]

    synthetic_data_features_drop = synthetic_data[selected_columns_features]

    # --- Drop performance-related columns ---
    original_data_performance_drop = original_data[selected_columns_performances]

    synthetic_data_performance_drop = synthetic_data[selected_columns_performances]

    # --- Combine back ---
    original_data_combined = pd.concat([original_data_features_drop, original_data_performance_drop], axis=1)
    synthetic_data_combined = pd.concat([synthetic_data_features_drop, synthetic_data_performance_drop], axis=1)

    # Encode numeric and categorical features
    num_cols = original_data_combined.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = original_data_combined.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(), cat_cols)
    ])

    original_encoded = preprocessor.fit_transform(original_data_combined)
    synthetic_encoded = preprocessor.transform(synthetic_data_combined)

    # --- UMAP ---
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42).fit(original_encoded)
    synthetic_umap = umap_model.transform(synthetic_encoded)

    # --- Final Assembly ---
    synthetic_df_plot = pd.DataFrame(synthetic_umap, columns=["UMAP1", "UMAP2"])
    synthetic_df_plot["Type"] = "Synthetic"

    interactive_df3 = pd.concat([
        synthetic_df_plot.reset_index(drop=True),
        synthetic_data_combined.reset_index(drop=True)
    ], axis=1)

    interactive_df3["Index"] = interactive_df3.index + 1 + len(original_data)
    return interactive_df3
