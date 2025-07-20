import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.manifold import trustworthiness
import umap

def get_interactive_df2_original(df):
    # Clean column names: strip and lowercase
    df.columns = df.columns.str.strip().str.lower()

    # Clean all categorical values: strip and lowercase
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()

    # Selected columns (performance space)
    selected_columns = ["clarity", "colour", "physical state", "initial ph", 
                        "appearance absorption value", "height foam (mm)", 
                        "cmc", "mildness"]
    df_performance_drop = df[selected_columns]

    # Encode numeric and categorical features
    num_cols = df_performance_drop.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df_performance_drop.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(), cat_cols)
    ])

    df_encoded = preprocessor.fit_transform(df_performance_drop)

    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    df_umap = umap_model.fit_transform(df_encoded)

    trust = trustworthiness(df_encoded, df_umap, n_neighbors=15)
    print(f"[UMAP2] Trustworthiness Score: {trust:.3f}")

    # Build combined UMAP dataframe
    df_plot = pd.DataFrame(df_umap, columns=["UMAP1", "UMAP2"])
    df_plot["Type"] = "Original"

    interactive_df2 = pd.concat([
        df_plot.reset_index(drop=True),
        df.reset_index(drop=True)
    ], axis=1)

    interactive_df2["Index"] = interactive_df2.index
    return interactive_df2


def get_interactive_df2_synthetic(original_data, synthetic_data):
    # Clean whitespace
    for df in [original_data, synthetic_data]:
        # Clean column names: strip and lowercase
        df.columns = df.columns.str.strip().str.lower()

        # Clean all categorical values: strip and lowercase
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            df[col] = df[col].astype(str).str.strip().str.lower()

    # Selected columns (performance space)
    selected_columns = ["clarity", "colour", "physical state", "initial ph", 
                        "appearance absorption value", "height foam (mm)", 
                        "cmc", "mildness"]
    original_data_performance_drop = original_data[selected_columns]

    synthetic_data_performance_drop = synthetic_data[selected_columns]


    # UMAP on numeric features only
    num_cols = original_data_performance_drop.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = original_data_performance_drop.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(), cat_cols)
    ])

    original_encoded = preprocessor.fit_transform(original_data_performance_drop)
    synthetic_encoded = preprocessor.transform(synthetic_data_performance_drop)

    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42).fit(original_encoded)
    synthetic_umap = umap_model.transform(synthetic_encoded)

    # Build combined UMAP dataframe
    synthetic_df_plot = pd.DataFrame(synthetic_umap, columns=["UMAP1", "UMAP2"])
    synthetic_df_plot["Type"] = "Synthetic"

    interactive_df2 = pd.concat([
        synthetic_df_plot.reset_index(drop=True),
        synthetic_data.reset_index(drop=True)
    ], axis=1)

    interactive_df2["Index"] = interactive_df2.index + 1 + len(original_data)
    return interactive_df2
