import pandas as pd

def clean_data(df):
    df = df.copy()

    # Clean column names: strip and lowercase
    df.columns = df.columns.str.strip().str.lower()

    # Clean all categorical values: strip and lowercase
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()

    return df

def transform_cmc(value):
    if value > 0.03:
        return '>0.03'
    else:
        return '<=0.03'

def simplify_categories(df):
    df = df.copy()

    # --- Clarity ---
    clarity_map = {
        'clear': 'clear',
        'turbid': 'turbid',
        'slightly turbid': 'clear',
    }
    df['clarity'] = df['clarity'].replace(clarity_map)

    # --- Colour ---
    colour_map = {
        'colourless': 'colourless',
        'white': 'coloured',
        'yellow': 'coloured',
        'grey' : 'coloured',
    }
    df['colour'] = df['colour'].replace(colour_map)
    
    return df

def split_surfactant_levels(df):
    # Normalize column values to lowercase for consistent comparison
    df = df.copy()

    # Filter based on your rules
    df_primary = df[
        (df['primary surfactant name'] != 'no') &
        (df['secondary surfactant name'] == 'no') &
        (df['tertiary surfactant name'] == 'no')
    ]

    df_secondary = df[
        (df['primary surfactant name'] != 'no') &
        (df['secondary surfactant name'] != 'no') &
        (df['tertiary surfactant name'] == 'no')
    ]

    df_tertiary = df[
        (df['primary surfactant name'] != 'no') &
        (df['secondary surfactant name'] != 'no') &
        (df['tertiary surfactant name'] != 'no')
    ]

    return df_primary, df_secondary, df_tertiary
