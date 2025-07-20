import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO

from umap_helpers.interactive_df1 import get_interactive_df1_original, get_interactive_df1_synthetic
from umap_helpers.interactive_df2 import get_interactive_df2_original, get_interactive_df2_synthetic
from umap_helpers.interactive_df3 import get_interactive_df3_original, get_interactive_df3_synthetic
from umap_helpers.interactive_df4 import get_interactive_df4_original, get_interactive_df4_synthetic
from umap_helpers.clean import clean_data, simplify_categories, split_surfactant_levels, transform_cmc
from umap_helpers.apply_ctgan import syn_primary_gen, syn_secondary_gen, syn_tertiary_gen
from umap_helpers.apply_gbm import syn_prediction
from umap_helpers.calculating_other_factors import add_calculated_columns

st.set_page_config(layout="wide")
st.title("Page 2: UMAP Exploration")
st.markdown("Explore surfactant data in feature space, performance space, and combined space.")

# Cached function to read Excel file
@st.cache_data
def load_excel(file):
    return pd.read_excel(file)

# File uploader
uploaded_file_original = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])

# Only proceed if a file is uploaded
if uploaded_file_original is not None:
    original_data = load_excel(uploaded_file_original)
    original_data = clean_data(original_data)
    original_data = simplify_categories(original_data)
    original_data['cmc'] = original_data['cmc'].apply(transform_cmc)
    st.session_state.uploaded_file_original = original_data
   
elif "uploaded_file_original" not in st.session_state:
    st.info("📂 Please upload an Excel file to proceed.")

if "uploaded_file_original" not in st.session_state:
    st.stop()

original_data = st.session_state.uploaded_file_original

# --- Cache Precomputed UMAPs ---
@st.cache_data
def compute_umap_data_original(original_data):
    df1 = get_interactive_df1_original(original_data)
    df2 = get_interactive_df2_original(original_data)
    df3 = get_interactive_df3_original(original_data)
    return df1, df2, df3

# --- Store UMAP results in session state if not already stored ---
if "interactive_df1_original" not in st.session_state:
    df1, df2, df3 = compute_umap_data_original(original_data)
    st.session_state.interactive_df1_original = df1
    st.session_state.interactive_df2_original = df2
    st.session_state.interactive_df3_original = df3

# --- Always work from session state copies ---
interactive_df1 = st.session_state.interactive_df1_original
interactive_df2 = st.session_state.interactive_df2_original
interactive_df3 = st.session_state.interactive_df3_original


# --- Synthetic Data Generator ---
@st.cache_data
def synthetic_data_gen(original_data):
    original_data = original_data.copy()
    df_primary, df_secondary, df_tertiary = split_surfactant_levels(original_data)
    df_syn_primary = syn_primary_gen(df_primary)
    df_syn_secondary = syn_secondary_gen(df_secondary)
    df_syn_tertiary = syn_tertiary_gen(df_tertiary)
    synthetic_df = syn_prediction(original_data, df_syn_primary, df_syn_secondary, df_syn_tertiary)
    return synthetic_df

# --- Button to Trigger Generation ---
run_synthetic_data_generation = st.button("Generate Synthetic Data")

if run_synthetic_data_generation and "synthetic_data" not in st.session_state:
    st.session_state.synthetic_data = synthetic_data_gen(original_data)
    
    # Compute UMAPs
    st.session_state.interactive_df1_syn = get_interactive_df1_synthetic(original_data, st.session_state.synthetic_data)
    st.session_state.interactive_df2_syn = get_interactive_df2_synthetic(original_data, st.session_state.synthetic_data)
    st.session_state.interactive_df3_syn = get_interactive_df3_synthetic(original_data, st.session_state.synthetic_data)

# --- Merge if synthetic exists ---
if "synthetic_data" in st.session_state:
    interactive_df1 = pd.concat([st.session_state.interactive_df1_original, st.session_state.interactive_df1_syn], ignore_index=True)
    interactive_df2 = pd.concat([st.session_state.interactive_df2_original, st.session_state.interactive_df2_syn], ignore_index=True)
    interactive_df3 = pd.concat([st.session_state.interactive_df3_original, st.session_state.interactive_df3_syn], ignore_index=True)

# --- Excel conversion utility ---
@st.cache_data
def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='FilteredData')
    return output.getvalue()

interactive_df1_columns = ["primary surfactant name", "secondary surfactant name", "tertiary surfactant name",
                          "primary surfactant level (%)", "secondary surfactant level (%)", "tertiary surfactant level (%)"]

interactive_df2_columns = ["clarity", "colour", "physical state", "initial ph", 
                        "appearance absorption value", "height foam (mm)", 
                        "cmc", "mildness"]

interactive_df3_columns = interactive_df1_columns + interactive_df2_columns


# Precompute UMAP axis ranges for fixed scale
fig1_umap1_range = [interactive_df1["UMAP1"].min()-1, interactive_df1["UMAP1"].max()+1]
fig1_umap2_range = [interactive_df1["UMAP2"].min()-1, interactive_df1["UMAP2"].max()+1]

fig2_umap1_range = [interactive_df2["UMAP1"].min()-1, interactive_df2["UMAP1"].max()+1]
fig2_umap2_range = [interactive_df2["UMAP2"].min()-1, interactive_df2["UMAP2"].max()+1]

fig3_umap1_range = [interactive_df3["UMAP1"].min()-1, interactive_df3["UMAP1"].max()+1]
fig3_umap2_range = [interactive_df3["UMAP2"].min()-1, interactive_df3["UMAP2"].max()+1]

# --- Filter Sidebar ---
st.sidebar.header("Filter Options")
if "filter_state" not in st.session_state:
    st.session_state.filter_state = {"type": "Original+Synthetic+New", "cat": {}, "num": {}}

# Add Apply Button to top
apply_filters = st.sidebar.button("\u2705 Apply Filters")

# Temporary state for new selections
# Initialize session state for radio button
if "type" not in st.session_state:
    st.session_state["type"] = st.session_state.filter_state.get("type", "Original+Synthetic+New")

# Render radio button with key to bind session state
temp_data_type = st.sidebar.radio( "Select Data Type", ["Original", "Synthetic", "New", "Original+Synthetic+New"], key="type")

cat_cols = ["primary surfactant name", "secondary surfactant name", "tertiary surfactant name", 
            "clarity", "colour", "physical state", "cmc", "height foam (mm)"]
num_cols = ["primary surfactant level (%)", "secondary surfactant level (%)", "tertiary surfactant level (%)",
            "initial ph", "appearance absorption value", "mildness"]

temp_cat_filters = {}
temp_num_filters = {}

df_filter = interactive_df3.copy()

for col in cat_cols:
    if col in df_filter.columns:
        options = sorted(df_filter[col].unique())
        # First-time default setting only
        if f"cat_{col}" not in st.session_state:
            st.session_state[f"cat_{col}"] = st.session_state.filter_state["cat"].get(col, options)
        temp_cat_filters[col] = st.sidebar.multiselect(col, options, key=f"cat_{col}")

for col in num_cols:
    if col in df_filter.columns:
        min_val, max_val = float(df_filter[col].min()), float(df_filter[col].max())
        if min_val != max_val:
            # First-time default setting only
            if f"num_{col}" not in st.session_state:
                st.session_state[f"num_{col}"] = st.session_state.filter_state["num"].get(col, (min_val, max_val))

            temp_num_filters[col] = st.sidebar.slider(col, min_val, max_val, step=0.01, key=f"num_{col}")
        else:
            st.sidebar.write(f"No value to adjust in `{col}`.")


# Update session state only when apply is pressed
if apply_filters:
    st.session_state.filter_state["type"] = temp_data_type
    for col in temp_cat_filters:
        st.session_state.filter_state["cat"][col] = temp_cat_filters[col]
    for col in temp_num_filters:
        st.session_state.filter_state["num"][col] = temp_num_filters[col]

# Apply all filters
if st.session_state.filter_state["type"] != "Original+Synthetic+New":
    df_filter = df_filter[df_filter["Type"] == st.session_state.filter_state["type"]]

for col, selected in st.session_state.filter_state["cat"].items():
    if selected:
        df_filter = df_filter[df_filter[col].isin(selected)]

for col, (low, high) in st.session_state.filter_state["num"].items():
    df_filter = df_filter[(df_filter[col] >= low) & (df_filter[col] <= high)]

indices = df_filter["Index"].tolist()
df1_filtered = interactive_df1[interactive_df1["Index"].isin(indices)]
df2_filtered = interactive_df2[interactive_df2["Index"].isin(indices)]
df3_filtered = interactive_df3[interactive_df3["Index"].isin(indices)]

# UMAP Plot 1: Feature Space
with st.expander("UMAP: Feature Space", expanded=True):
    st.subheader("UMAP: Feature Space")

    if not df1_filtered.empty:
        # Let user optionally select a feature for color coding
        feature_columns = ["primary surfactant name", "secondary surfactant name", "tertiary surfactant name",
                            "primary surfactant level (%)", "secondary surfactant level (%)", "tertiary surfactant level (%)"]
        
        # Set default only if not already in session state
        if "color_feature_selected" not in st.session_state:
            st.session_state.color_feature_selected = "Type"  # store as a string

        def feature_color():
            st.session_state.color_feature_selected = st.session_state.color_feature_1

        # Get the current selection index
        default_index_feature = (["Type"] + feature_columns).index(st.session_state.color_feature_selected)

        color_feature = st.selectbox("Color by (optional):", ["Type"] + feature_columns,
                                    index=default_index_feature, key="color_feature_1",
                                    on_change=feature_color)
        
        custom_colors = [c for c in px.colors.qualitative.Plotly if c not in ['#EF553B', '#636EFA']]

        # Setup color parameters
        if color_feature == "Type":
            fig1 = px.scatter(
                df1_filtered,
                x="UMAP1",
                y="UMAP2",
                color="Type",
                hover_data=interactive_df1_columns,
                color_discrete_map={"Original": "#4A90E2", "Synthetic": "#E74C3C", "New": "#553500"}
            )
        
        elif color_feature in ["primary surfactant name", "secondary surfactant name", "tertiary surfactant name"]:
            fig1 = px.scatter(
                df1_filtered,
                x="UMAP1",
                y="UMAP2",
                color=color_feature,
                hover_data=interactive_df1_columns,
                color_discrete_sequence=custom_colors
            )

        else:
            fig1 = px.scatter(
                df1_filtered,
                x="UMAP1",
                y="UMAP2",
                color=color_feature,
                color_continuous_scale="plasma",
                hover_data=interactive_df1_columns
            )

        fig1.update_traces(marker=dict(size=6, opacity=0.7))
        fig1.update_layout(
            width=900,
            height=600,
            xaxis=dict(range=fig1_umap1_range),
            yaxis=dict(range=fig1_umap2_range),
            title=f"UMAP Colored by '{color_feature}'"
        )
        st.plotly_chart(fig1, use_container_width=True, key="fig1")
    else:
        st.warning("No data to show in Feature Space UMAP.")


# UMAP Plot 2: Performance Space
with st.expander("UMAP: Performance Space", expanded=True):
    st.subheader("UMAP: Performance Space")

    if not df2_filtered.empty:
        performance_columns = ["clarity", "colour", "physical state", "initial ph", 
                            "appearance absorption value", "height foam (mm)", 
                            "cmc", "mildness"]
        
        # Set default only if not already in session state
        if "color_performance_selected" not in st.session_state:
            st.session_state.color_performance_selected = "Type"  # store as a string

        def performance_color():
            st.session_state.color_performance_selected = st.session_state.color_feature_2

        # Get the current selection index
        default_index_performance = (["Type"] + performance_columns).index(st.session_state.color_performance_selected)

        # Display the selectbox with the default index
        color_performance = st.selectbox("Color by (optional):", ["Type"] + performance_columns, 
                                    index=default_index_performance, key="color_feature_2", 
                                    on_change = performance_color)
        
        custom_colors = [c for c in px.colors.qualitative.Plotly if c not in ['#EF553B', '#636EFA']]

        if color_performance == "Type":
            fig2 = px.scatter(
                df2_filtered,
                x="UMAP1",
                y="UMAP2",
                color="Type",
                hover_data=interactive_df2_columns,
                color_discrete_map={"Original": "#4A90E2", "Synthetic": "#E74C3C", "New": "#553500"}
            )
        
        elif color_performance in ["clarity", "colour", "physical state", "cmc"]:
            fig2 = px.scatter(
                df2_filtered,
                x="UMAP1",
                y="UMAP2",
                color=color_performance,
                hover_data=interactive_df2_columns,
                color_discrete_sequence=custom_colors
            )

        else:
            fig2 = px.scatter(
                df2_filtered,
                x="UMAP1",
                y="UMAP2",
                color=color_performance,
                color_continuous_scale="plasma",
                hover_data=interactive_df2_columns
            )

        fig2.update_traces(marker=dict(size=6, opacity=0.7))
        fig2.update_layout(
            width=900, height=600,
            xaxis=dict(range=fig2_umap1_range),
            yaxis=dict(range=fig2_umap2_range),
            title=f"UMAP Colored by '{color_performance}'"
        )
        st.plotly_chart(fig2, use_container_width=True, key="fig2")
    else:
        st.warning("No data to show in Performance Space UMAP.")


# UMAP Plot 3: Combined Feature + Performance
with st.expander("UMAP: Combined Feature + Performance Space", expanded=True):
    st.subheader("UMAP: Combined Feature + Performance Space")

    if not df3_filtered.empty:
        feature_performance_columns = ["primary surfactant name", "secondary surfactant name", "tertiary surfactant name",
                                        "primary surfactant level (%)", "secondary surfactant level (%)", "tertiary surfactant level (%)",
                                        "clarity", "colour", "physical state", "initial ph", 
                                        "appearance absorption value", "height foam (mm)", 
                                        "cmc", "mildness"]
        
        # Set default only if not already in session state
        if "color_feature_performance_selected" not in st.session_state:
            st.session_state.color_feature_performance_selected = "Type"  # store as a string

        def combine_color():
            st.session_state.color_feature_performance_selected = st.session_state.color_feature_3

        # Get the current selection index
        default_index_feature_performance = (["Type"] + feature_performance_columns).index(st.session_state.color_feature_performance_selected)

        color_feature_performance = st.selectbox("Color by (optional):", ["Type"] + feature_performance_columns, 
                                       index=default_index_feature_performance, key="color_feature_3",
                                       on_change=combine_color)
        
        custom_colors = [c for c in px.colors.qualitative.Plotly if c not in ['#EF553B', '#636EFA']]

        if color_feature_performance == "Type":
            fig3 = px.scatter(
                df3_filtered,
                x="UMAP1",
                y="UMAP2",
                color="Type",
                hover_data=interactive_df3_columns,
                color_continuous_scale="plasma",
                color_discrete_map={"Original": "#4A90E2", "Synthetic": "#E74C3C", "New": "#553500"}
            )
        
        elif color_feature_performance in ["primary surfactant name", "secondary surfactant name", "tertiary surfactant name", "clarity", "colour", "physical state", "cmc"]:
            fig3 = px.scatter(
                df3_filtered,
                x="UMAP1",
                y="UMAP2",
                color=color_feature_performance,
                hover_data=interactive_df3_columns,
                color_discrete_sequence=custom_colors
            )

        else:
            fig3 = px.scatter(
                df3_filtered,
                x="UMAP1",
                y="UMAP2",
                color=color_feature_performance,
                hover_data=interactive_df3_columns
            )

        fig3.update_traces(marker=dict(size=6, opacity=0.7))
        fig3.update_layout(
            width=900, height=600,
            xaxis=dict(range=fig3_umap1_range),
            yaxis=dict(range=fig3_umap2_range),
            title=f"UMAP Colored by '{color_feature_performance}'"
        )
        st.plotly_chart(fig3, use_container_width=True, key="fig3")
    else:
        st.warning("No data to show in Combined UMAP.")


# UMAP Plot 4: Custom UMAP
with st.expander("UMAP: Custom Selected Variables", expanded=True):
    st.subheader("UMAP: Custom Selected Variables")

    available_cols = interactive_df3_columns

    if "custom" not in st.session_state:
        st.session_state["custom"] = st.session_state.get("custom_umap_features", [])

    selected_cols = st.multiselect(
        "Select variables for Custom UMAP projection:",
        available_cols,
        key="custom"
    )

    custom_columns = [col for col in selected_cols if col not in ["UMAP1", "UMAP2", "Type", "Index"]]

    # Set default only if not already in session state
    if "color_custom_selected" not in st.session_state:
        st.session_state.color_custom_selected = "Type"  # store as a string

    def custom_color():
        st.session_state.color_custom_selected = st.session_state.color_feature_4

    # Get the current selection index
    default_index_custom = (["Type"] + custom_columns).index(st.session_state.color_custom_selected)

    color_custom = st.selectbox("Color by (optional):", ["Type"] + custom_columns, 
                                index=default_index_custom, key="color_feature_4",
                                on_change=custom_color)

    run_custom_umap = st.button("Generate Custom UMAP")

    if run_custom_umap and selected_cols and color_custom:
        st.session_state["custom_umap_features"] = selected_cols
        original_data = original_data.copy()
        original_data_custom = original_data[selected_cols].copy()

        st.session_state.interactive_df4_original = get_interactive_df4_original(original_data_custom)
        custom_df = st.session_state.interactive_df4_original

        if "synthetic_data" in st.session_state:
            synthetic_data = st.session_state.synthetic_data
            synthetic_data_custom = synthetic_data[selected_cols].copy()
            custom_df_synthetic = get_interactive_df4_synthetic(original_data_custom, synthetic_data_custom)
            custom_df = pd.concat([st.session_state.interactive_df4_original, custom_df_synthetic])

        custom_umap1_range = [custom_df["UMAP1"].min() - 1, custom_df["UMAP1"].max() + 1]
        custom_umap2_range = [custom_df["UMAP2"].min() - 1, custom_df["UMAP2"].max() + 1]

        st.markdown(f"**UMAP based on: {', '.join(selected_cols)}**")

        df4_filtered = custom_df[custom_df["Index"].isin(indices)]
        custom_colors = [c for c in px.colors.qualitative.Plotly if c not in ['#EF553B', '#636EFA']]

        if not df4_filtered.empty:
            # Color selection for Custom UMAP
            if color_custom == "Type":
                fig4 = px.scatter(
                    df4_filtered,
                    x="UMAP1",
                    y="UMAP2",
                    color="Type",
                    hover_data=selected_cols,
                    color_discrete_map={"Original": "#4A90E2", "Synthetic": "#E74C3C", "New": "#553500"}
                )
            
            elif color_custom in ["primary surfactant name", "secondary surfactant name", "tertiary surfactant name", "clarity", "colour", "physical state", "cmc"]:
                fig4 = px.scatter(
                    df4_filtered,
                    x="UMAP1",
                    y="UMAP2",
                    color=color_custom,
                    hover_data=selected_cols,
                    color_discrete_sequence=custom_colors
                )

            else:
                fig4 = px.scatter(
                    df4_filtered,
                    x="UMAP1",
                    y="UMAP2",
                    color=color_custom,
                    color_continuous_scale="plasma",
                    hover_data=selected_cols
                )

            fig4.update_traces(marker=dict(size=6, opacity=0.7))
            fig4.update_layout(
                width=900,
                height=600,
                xaxis=dict(range=custom_umap1_range),
                yaxis=dict(range=custom_umap2_range),
                title=f"UMAP Colored by '{color_custom}'"
            )
            st.plotly_chart(fig4, use_container_width=True, key="fig4")
        else:
            st.warning("No data to show in Custom UMAP.")
    elif run_custom_umap and not selected_cols:
        st.warning("Please select at least one variable to generate the Custom UMAP.")


with st.expander("Download Filtered Data", expanded=True):
    # Define column orders
    columns_button = [
        "primary surfactant name", "secondary surfactant name", "tertiary surfactant name",
        "primary surfactant level (%)", "secondary surfactant level (%)", "tertiary surfactant level (%)",
        "primary surfactant active (%)", "secondary surfactant active (%)", "tertiary surfactant active (%)",
        "total surfactant level (%)", "water (g)", "primary surfactant (g)", "secondary surfactant (g)", "tertiary surfactant (g)",
        "sodium benzoate (g)", "primary surfactant ratio", "secondary surfactant ratio", "tertiary surfactant ratio",
        "clarity", "colour", "physical state", "initial ph", "appearance absorption value", "height foam (mm)",
        "cmc", "mildness"
    ]

    df_filtered = df3_filtered.copy()
    df_filtered_with_calcs = add_calculated_columns(df3_filtered)

    df_filtered_with_calcs = df_filtered_with_calcs[columns_button]
    n_rows, n_cols = df_filtered_with_calcs.shape

    st.success(f"Filtered data has {n_rows} rows and {n_cols} columns.")
    excel_data = convert_df_to_excel(df_filtered_with_calcs)
    st.download_button(
        label="Click to Download Excel with Calculations",
        data=excel_data,
        file_name="filtered_data_with_calculations.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
        
