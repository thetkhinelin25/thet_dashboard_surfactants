
def add_calculated_columns(df):
    # Step 1: Total Surfactant Level (%)
    df["total surfactant level (%)"] = (
        df.get("primary surfactant level (%)", 0) +
        df.get("secondary surfactant level (%)", 0) +
        df.get("tertiary surfactant level (%)", 0)
    )

    # --- Primary Surfactant Active (%) ---
    df["primary surfactant active (%)"] = df["primary surfactant name"].apply(
        lambda x: 24 if x == "adinol ct24" else (27 if x == "sles" else 30)
    )

    # --- Secondary Surfactant Active (%) ---
    df["secondary surfactant active (%)"] = df["secondary surfactant name"].apply(
        lambda x: 30 if x == "cab30" else 0
    )

    # --- Tertiary Surfactant Active (%) ---
    df["tertiary surfactant active (%)"] = df["tertiary surfactant name"].apply(
        lambda x: 50.63 if x == "decyl glucoside" else 0
    )

    # Step 2: Surfactant (g) for each level
    def calc_grams(level, active):
        try:
            return 70 / (active / level)
        except ZeroDivisionError:
            return 0

    if "primary surfactant level (%)" in df.columns:
        df["primary surfactant (g)"] = df.apply(
            lambda row: calc_grams(row["primary surfactant level (%)"], row["primary surfactant active (%)"]),
            axis=1
        )

    if "secondary surfactant level (%)" in df.columns:
        df["secondary surfactant (g)"] = df.apply(
            lambda row: calc_grams(row["secondary surfactant level (%)"], row["secondary surfactant active (%)"]),
            axis=1
        )

    if "tertiary surfactant level (%)" in df.columns:
        df["tertiary surfactant (g)"] = df.apply(
            lambda row: calc_grams(row["tertiary surfactant level (%)"], row["tertiary surfactant active (%)"]),
            axis=1
        )

    # Step 3: Sodium Benzoate (g)
    df["sodium benzoate (g)"] = 0.7

    # Step 4: Surfactant Ratios
    if "primary surfactant level (%)" in df.columns:
        df["primary surfactant ratio"] = df["primary surfactant level (%)"] / df["total surfactant level (%)"]

    if "secondary surfactant level (%)" in df.columns:
        df["secondary surfactant ratio"] = df["secondary surfactant level (%)"] / df["total surfactant level (%)"]

    if "tertiary surfactant level (%)" in df.columns:
        df["tertiary surfactant ratio"] = df["tertiary surfactant level (%)"] / df["total surfactant level (%)"]
    
    # Step 5: Water (g) = 70 - (sum of surfactants in g)
    df["water (g)"] = 70 - (
        df["primary surfactant (g)"] +
        df["secondary surfactant (g)"] +
        df["tertiary surfactant (g)"]
    )

    return df