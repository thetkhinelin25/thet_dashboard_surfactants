import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.manifold import trustworthiness
import umap

def get_interactive_df1_original(df):
    # Clean column names: strip and lowercase
    df.columns = df.columns.str.strip().str.lower()

    # Clean all categorical values: strip and lowercase
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()

    # Selected columns (separately because they differ)
    selected_columns = ["primary surfactant name", "secondary surfactant name", "tertiary surfactant name",
                        "primary surfactant level (%)", "secondary surfactant level (%)", "tertiary surfactant level (%)"]
    df_features_drop = df[selected_columns]

    # Identify column types
    num_cols = df_features_drop.select_dtypes(include='number').columns.tolist()
    cat_cols = df_features_drop.select_dtypes(include='object').columns.tolist()

    # One-hot + standard scaling
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(), cat_cols)
    ])

    df_encoded = preprocessor.fit_transform(df_features_drop)

    # UMAP
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    df_umap = umap_model.fit_transform(df_encoded)

    trust = trustworthiness(df_encoded, df_umap, n_neighbors=15)
    print(f"[UMAP1] Trustworthiness Score: {trust:.3f}")

    # Combine UMAP coordinates with original rows
    df_plot = pd.DataFrame(df_umap, columns=["UMAP1", "UMAP2"])
    df_plot["Type"] = "Original"

    interactive_df1 = pd.concat([
        df_plot.reset_index(drop=True),
        df.reset_index(drop=True)
    ], axis=1)

    interactive_df1["Index"] = interactive_df1.index
    return interactive_df1


def get_interactive_df1_synthetic(original_data, synthetic_data):
    # Clean string columns
    for df in [original_data, synthetic_data]:
        # Clean column names: strip and lowercase
        df.columns = df.columns.str.strip().str.lower()

        # Clean all categorical values: strip and lowercase
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            df[col] = df[col].astype(str).str.strip().str.lower()

    # Selected columns (separately because they differ)
    selected_columns = ["primary surfactant name", "secondary surfactant name", "tertiary surfactant name",
                        "primary surfactant level (%)", "secondary surfactant level (%)", "tertiary surfactant level (%)"]
    original_data_features_drop = original_data[selected_columns]

    synthetic_data_features_drop = synthetic_data[selected_columns]

    # Identify column types
    num_cols = original_data_features_drop.select_dtypes(include='number').columns.tolist()
    cat_cols = original_data_features_drop.select_dtypes(include='object').columns.tolist()

    # One-hot + standard scaling
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(), cat_cols)
    ])

    original_encoded = preprocessor.fit_transform(original_data_features_drop)
    synthetic_encoded = preprocessor.transform(synthetic_data_features_drop)

    # UMAP
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42).fit(original_encoded)
    synthetic_umap = umap_model.transform(synthetic_encoded)

    # Combine UMAP coordinates with original rows
    synthetic_df_plot = pd.DataFrame(synthetic_umap, columns=["UMAP1", "UMAP2"])
    synthetic_df_plot["Type"] = "Synthetic"

    interactive_df1 = pd.concat([
        synthetic_df_plot.reset_index(drop=True),
        synthetic_data.reset_index(drop=True)
    ], axis=1)

    interactive_df1["Index"] = interactive_df1.index + 1 + len(original_data)
    return interactive_df1
