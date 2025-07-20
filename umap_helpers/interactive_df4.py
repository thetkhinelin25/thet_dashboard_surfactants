import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.manifold import trustworthiness
import umap

def get_interactive_df4_original(df):
    # Clean column names: strip and lowercase
    df.columns = df.columns.str.strip().str.lower()

    # Clean all categorical values: strip and lowercase
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()

    # --- Select columns ---
    num_cols_all = ["primary surfactant level (%)", "secondary surfactant level (%)", "tertiary surfactant level (%)",
                "initial ph", "appearance absorption value", "mildness"]
    cat_cols_non_order_all = ["primary surfactant name", "secondary surfactant name", "tertiary surfactant name",
                              "clarity", "colour", "physical state", "cmc", "height foam (mm)"]

    # Filter only columns that exist in the DataFrame
    num_cols = [col for col in num_cols_all if col in df.columns]
    cat_cols_non_order = [col for col in cat_cols_non_order_all if col in df.columns]

    # Define preprocessor
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat_non_order", OneHotEncoder(), cat_cols_non_order)
    ])

    df_encoded = preprocessor.fit_transform(df)

    # --- UMAP ---
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    df_umap = umap_model.fit_transform(df_encoded)

    trust = trustworthiness(df_encoded, df_umap, n_neighbors=15)
    print(f"[UMAP4] Trustworthiness Score: {trust:.3f}")

    # --- Final Assembly ---
    df_plot = pd.DataFrame(df_umap, columns=["UMAP1", "UMAP2"])
    df_plot["Type"] = "Original"

    interactive_df4 = pd.concat([
        df_plot.reset_index(drop=True),
        df.reset_index(drop=True)
    ], axis=1)

    interactive_df4["Index"] = interactive_df4.index
    return interactive_df4


def get_interactive_df4_synthetic(original_data_custom, synthetic_data_custom):

    for df in [original_data_custom, synthetic_data_custom]:
        # Clean column names: strip and lowercase
        df.columns = df.columns.str.strip().str.lower()

        # Clean all categorical values: strip and lowercase
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            df[col] = df[col].astype(str).str.strip().str.lower()

    # --- Select columns ---
    num_cols_all = ["primary surfactant level (%)", "secondary surfactant level (%)", "tertiary surfactant level (%)",
                "initial ph", "appearance absorption value", "mildness"]
    cat_cols_non_order_all = ["primary surfactant name", "secondary surfactant name", "tertiary surfactant name",
                              "clarity", "colour", "physical state", "cmc", "height foam (mm)"]

    # Filter only columns that exist in the DataFrame
    num_cols = [col for col in num_cols_all if col in original_data_custom.columns]
    cat_cols_non_order = [col for col in cat_cols_non_order_all if col in original_data_custom.columns]

    # Define preprocessor
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat_non_order", OneHotEncoder(), cat_cols_non_order)
    ])

    original_encoded = preprocessor.fit_transform(original_data_custom)
    synthetic_encoded = preprocessor.transform(synthetic_data_custom)

    # --- UMAP ---
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42).fit(original_encoded)
    synthetic_umap = umap_model.transform(synthetic_encoded)

    # --- Final Assembly ---
    synthetic_df_plot = pd.DataFrame(synthetic_umap, columns=["UMAP1", "UMAP2"])
    synthetic_df_plot["Type"] = "Synthetic"

    interactive_df4 = pd.concat([
        synthetic_df_plot.reset_index(drop=True),
        synthetic_data_custom.reset_index(drop=True)
    ], axis=1)

    interactive_df4["Index"] = interactive_df4.index + 1 + len(original_data_custom)
    return interactive_df4


