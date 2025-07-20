
import pandas as pd
import numpy as np
import random
import torch
from sdv.tabular import CTGAN
from sklearn.preprocessing import LabelEncoder


# -- For Generating Prinary Surfactant Blends -- 
def syn_primary_gen(df_primary):
    SEED = 9999
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Drop unused columns
    df_primary = df_primary[['primary surfactant name', 'secondary surfactant name', 'tertiary surfactant name',
                             'primary surfactant level (%)', 'secondary surfactant level (%)', 'tertiary surfactant level (%)']]

    # ---- Step 2: Encode categorical columns ----
    categorical_columns = df_primary.select_dtypes(include='object').columns.tolist()
    encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df_primary[col] = le.fit_transform(df_primary[col].astype(str))
        encoders[col] = le

    # ---- Step 3: Progressive Generation ----
    original_data = df_primary.copy()
    all_generated = []
    target = 100  # Number of unique samples per round

    def generate_unique_samples(model, original_data, target, max_attempts=5):
        oversample_factor = 3
        attempts = 0
        while attempts < max_attempts:
            synthetic = model.sample(target * oversample_factor)
            synthetic = synthetic.drop_duplicates(subset=original_data.columns.tolist()).reset_index(drop=True)
            if len(synthetic) >= target:
                return synthetic.iloc[:target]
            else:
                oversample_factor += 1
                attempts += 1
        return synthetic  # Return as many as possible

    for i in range(3):  # 3 iterations
        print(f"\n🚀 Iteration {i+1}: Training CTGAN and generating unique samples...")

        model = CTGAN(
            epochs=300,
            generator_dim=[30, 15],
            discriminator_dim=[64, 32],
            generator_lr=0.0001,
            discriminator_lr=0.0001,
            cuda=False
        )

        model.fit(original_data)
        new_samples = generate_unique_samples(model, original_data, target)

        # Add to combined dataset
        all_generated.append(new_samples)
        original_data = pd.concat([original_data, new_samples], ignore_index=True)

    # ---- Step 4: Decode categorical values back to original ----
    final_synthetic = pd.concat(all_generated, ignore_index=True)
    for col in categorical_columns:
        final_synthetic[col] = encoders[col].inverse_transform(final_synthetic[col].round().astype(int))

    # ---- Step 5: Final de-duplication (optional, based on key features) ----
    final_synthetic = final_synthetic.drop_duplicates(subset=[
        "primary surfactant name",
        "secondary surfactant name",
        "tertiary surfactant name",
        "primary surfactant level (%)",
        "secondary surfactant level (%)",
        "tertiary surfactant level (%)"
    ]).reset_index(drop=True)

    return final_synthetic



# -- For Generating Secondary Surfactant Blends -- 
def syn_secondary_gen(df_secondary):
    SEED = 9999
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Drop unused or irrelevant columns
    df_secondary = df_secondary[['primary surfactant name', 'secondary surfactant name', 'tertiary surfactant name',
                                'primary surfactant level (%)', 'secondary surfactant level (%)', 'tertiary surfactant level (%)']]

    # ---- Step 2: Encode categorical columns ----
    categorical_columns = df_secondary.select_dtypes(include='object').columns.tolist()
    encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df_secondary[col] = le.fit_transform(df_secondary[col].astype(str))
        encoders[col] = le

    # ---- Step 3: Define filter and generator helper ----
    level_cols = ["primary surfactant level (%)", "secondary surfactant level (%)", "tertiary surfactant level (%)"]
    target = 100

    def generate_unique_filtered_samples(model, original_data, target, level_cols, max_attempts=100):
        oversample_factor = 3
        attempts = 0
        while attempts < max_attempts:
            synthetic = model.sample(target * oversample_factor)

            # Filter: Total surfactant level between 14.5 and 15
            synthetic["total surfactant level"] = synthetic[level_cols].sum(axis=1)
            synthetic = synthetic[(synthetic["total surfactant level"] > 14.5) &
                                (synthetic["total surfactant level"] < 15)]

            synthetic.drop(columns=["total surfactant level"], inplace=True)

            # Drop duplicates
            synthetic = synthetic.drop_duplicates(subset=original_data.columns.tolist()).reset_index(drop=True)

            if len(synthetic) >= target:
                return synthetic.iloc[:target]
            else:
                oversample_factor += 1
                attempts += 1
        return synthetic

    # ---- Step 4: Progressive generation ----
    original_data = df_secondary.copy()
    all_generated = []

    for i in range(3):  # 3 rounds of generation
        print(f"\n🚀 Iteration {i+1}: Training CTGAN and generating filtered unique samples...")

        model = CTGAN(
            epochs=300,
            generator_dim=[30, 15],
            discriminator_dim=[64, 32],
            generator_lr=0.0001,
            discriminator_lr=0.0001,
            cuda=False
        )

        model.fit(original_data)
        new_samples = generate_unique_filtered_samples(model, original_data, target, level_cols)

        # Append and update original data
        all_generated.append(new_samples)
        original_data = pd.concat([original_data, new_samples], ignore_index=True)

    # ---- Step 5: Decode categorical variables ----
    final_synthetic = pd.concat(all_generated, ignore_index=True)
    for col in categorical_columns:
        final_synthetic[col] = encoders[col].inverse_transform(final_synthetic[col].round().astype(int))

    # ---- Step 6: Final deduplication (optional) ----
    final_synthetic = final_synthetic.drop_duplicates(subset=[
        "primary surfactant name",
        "secondary surfactant name",
        "tertiary surfactant name",
        "primary surfactant level (%)",
        "secondary surfactant level (%)",
        "tertiary surfactant level (%)"
    ]).reset_index(drop=True)

    return final_synthetic



# -- For Generating Tertiary Surfactant Blends -- 
def syn_tertiary_gen(df_tertiary):
    SEED = 9999
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    df_tertiary = df_tertiary[['primary surfactant name', 'secondary surfactant name', 'tertiary surfactant name',
                            'primary surfactant level (%)', 'secondary surfactant level (%)', 'tertiary surfactant level (%)']]

    # ---- Step 2: Encode categorical columns ----
    categorical_columns = df_tertiary.select_dtypes(include='object').columns.tolist()
    encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df_tertiary[col] = le.fit_transform(df_tertiary[col].astype(str))
        encoders[col] = le

    # ---- Step 3: Define helper function for generation with filtering ----
    level_cols = ['primary surfactant level (%)', 'secondary surfactant level (%)', 'tertiary surfactant level (%)']
    target = 100
    max_attempts = 5

    def generate_unique_filtered_samples(model, original_data, target, level_cols, max_attempts=100):
        oversample_factor = 3
        attempts = 0
        while attempts < max_attempts:
            synthetic = model.sample(target * oversample_factor)

            # Filter: Total surfactant level between 14.5 and 15
            synthetic["total surfactant level"] = synthetic[level_cols].sum(axis=1)
            synthetic = synthetic[(synthetic["total surfactant level"] > 14.5) &
                                (synthetic["total surfactant level"] < 15)]
            synthetic.drop(columns=["total surfactant level"], inplace=True)

            # Drop duplicates
            synthetic = synthetic.drop_duplicates(subset=original_data.columns.tolist()).reset_index(drop=True)

            if len(synthetic) >= target:
                return synthetic.iloc[:target]
            else:
                oversample_factor += 1
                attempts += 1
        return synthetic

    # ---- Step 4: Progressive generation ----
    original_data = df_tertiary.copy()
    all_generated = []

    for i in range(3):  # 3 rounds of 100 each
        print(f"\n🚀 Iteration {i+1}: Training CTGAN and generating filtered unique samples...")

        model = CTGAN(
            epochs=300,
            generator_dim=[30, 15],
            discriminator_dim=[64, 32],
            generator_lr=0.0001,
            discriminator_lr=0.0001,
            cuda=False
        )

        model.fit(original_data)
        new_samples = generate_unique_filtered_samples(model, original_data, target, level_cols)

        all_generated.append(new_samples)
        original_data = pd.concat([original_data, new_samples], ignore_index=True)

    # ---- Step 5: Decode categorical columns ----
    final_synthetic = pd.concat(all_generated, ignore_index=True)
    for col in categorical_columns:
        final_synthetic[col] = encoders[col].inverse_transform(final_synthetic[col].round().astype(int))

    # ---- Step 6: Final deduplication (optional) ----
    final_synthetic = final_synthetic.drop_duplicates(subset=[
        "primary surfactant name",
        "secondary surfactant name",
        "tertiary surfactant name",
        "primary surfactant level (%)",
        "secondary surfactant level (%)",
        "tertiary surfactant level (%)"
    ]).reset_index(drop=True)

    return final_synthetic