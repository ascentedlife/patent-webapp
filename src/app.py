import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
import wordcloud

import model

FILE = os.path.dirname(__file__)
LABEL_MAP = {1: "Product Innovation", 0: "Process Innovation"}
TECH_MAP = {
    "SPV": "Solar Photovoltaic",
    "SC": "Supercapacitors",
    "LIB": "Lithium-Ion Batteries",
    "LC": "Lead Acid Batteries",
    "FW": "Flywheels",
    "Wind": "Wind",
}
FIXED_COLORS = {
    tech: px.colors.qualitative.Plotly[i]
    for i, tech in enumerate(sorted(TECH_MAP.values()))
}
STOPWORDS = wordcloud.STOPWORDS

with open(os.path.join(FILE, "stopwords"), encoding="utf8") as f:
    STOPWORDS.update(f.read().splitlines())


@st.cache_data
def load_data(file_path):
    """Load and cache the dataset."""
    claims_data = pd.read_csv(file_path)
    claims_data["Labels"] = claims_data["Labels"].map(LABEL_MAP)
    claims_data["Tech"] = claims_data["Tech"].map(TECH_MAP)
    return claims_data


@st.cache_data
def options(claims_data):
    """Get unique values for filters."""
    techs = claims_data["Tech"].unique()
    sources = claims_data["Patent Source"].unique()
    labels = claims_data["Labels"].unique()
    return techs, sources, labels


@st.cache_data
def sample_data(data, n=10):
    """Return a random sample of the data."""
    return data.sample(n)


@st.cache_data
def generate_wordcloud(text, mask, max_words=200):
    """Generate a word cloud from the given text and mask."""
    wcloud = wordcloud.WordCloud(
        width=1200,
        height=600,
        mask=mask,
        max_words=max_words,
        background_color="white",
        contour_color="black",
        contour_width=0,
        stopwords=STOPWORDS,
    ).generate(text)
    return wcloud


@st.cache_data
def load_mask(mask_path):
    """Load mask image for word cloud."""
    return np.array(Image.open(mask_path))


@st.cache_resource
def load_model(model_path):
    """Load the pre-trained model."""
    return model.PatentClaimClassifier(model_path)


def plot_bar_chart(df):
    """Plot a bar chart with consistent colors."""
    fig = px.bar(
        df,
        x="Product/Process Ratio",
        y="Tech",
        orientation="h",
        title="Product/Process Innovation Ratio by Technology",
        labels={"Product/Process Ratio": "Product/Process Ratio", "Tech": "Technology"},
        hover_data={"Product/Process Ratio": ":.2%"},
        color="Tech",
        color_discrete_map=FIXED_COLORS,
    )
    fig.update_layout(xaxis=dict(showgrid=True, gridcolor="LightGrey"))
    return fig


def display_summary_statistics(df):
    """Display summary statistics in a more visually appealing way."""
    total_rows = len(df)
    total_product_innovations = len(df[df["Labels"] == "Product Innovation"])
    total_process_innovations = len(df[df["Labels"] == "Process Innovation"])
    avg_text_length = df["Text"].apply(len).mean()

    st.write(f"**Total Rows:** {total_rows:,}")
    st.write(f"**Total Product Innovations:** {total_product_innovations:,}")
    st.write(f"**Total Process Innovations:** {total_process_innovations:,}")
    st.write(f"**Average Number of Characters in Text Column:** {avg_text_length:,.2f}")


def main():
    """Main Streamlit application."""
    st.set_page_config(layout="wide")
    # Load model and data
    xlnet = load_model(model_path="models/xlnet")
    claims_data = load_data(file_path="data/claims.csv")
    techs, sources, labels = options(claims_data)
    mask = load_mask("images/wordcloud_mask.jpeg")

    # Introduction
    st.title("Comparing Product-Process Innovation Share Across Technologies")

    overview_col, feature_col = st.columns(2)

    with overview_col:
        st.markdown("""
            ## :blue[Overview]
            #### :green[What is the purpose of the project?]

            Patents are key representations of technological innovations. Identifying patent claims to be either an innovation in product or process is a key step in technology policy research. This project aims to automate this classification process using a large language model (LLM). After assessing LLM options, the optimal trained LLM was used to classify a large dataset of patent claims. This webapp presents the results.

            #### :green[What is product and process innovation?]
            - **Product Innovation**: The development of new or improved products or services that differ significantly from existing products or services.
            - **Process Innovation**: The implementation of new or significantly improved production or delivery methods.
            """)

    with feature_col:
        st.markdown("""
            ## :blue[Features]
            - **Patent Claim Classifier**: Try out the LLM yourself! Classify a patent claim using a pre-trained large language model.
            - **Word Cloud**: Identify trends and focus areas within the patent data. This can reveal which topics or concepts are most prevalent in different technology categories.
            - **Analysis of Innovation Types**: Calculate and visuale the ratio between product and process innovations across different technology categories.
            - **Data Exploration and Visualization**: Explore the dataset used in this project, which includes various columns such as publication number, title, claim type, claim text, technology category, patent source, and classification labels.
            """)

    st.text("")
    st.text("")

    # Patent Claim Classifier
    with st.container(border=True):
        st.header(":briefcase: :rainbow[Patent Claim Classifier]")
        st.markdown(
            "The model classifies patent claims to be either a product or process innovation."
        )
        patent_claim = st.text_area(
            "Enter your patent claim here:",
            placeholder="Example: An implantable medical device comprising: control electronics for delivering therapy and/or monitoring physiological signals, the control electronics comprising: a processor...",
        )

        if st.button("Classify"):
            if patent_claim:
                prediction, certainty = xlnet.predict(patent_claim)
                st.success(
                    f"""
                    **Prediction**: {LABEL_MAP[prediction]}\n
                    **Certainty**: {certainty*100:.2f}%
                    """
                )

    # Word Cloud
    with st.container(border=True):
        st.header(":cloud: :rainbow[Word Cloud]")
        st.markdown(
            "Visualize the most common words in patent claims based on selected filters."
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            tech_filter = st.selectbox(
                "Select Technology", options=["All"] + list(techs)
            )
        with col2:
            source_filter = st.selectbox(
                "Select Patent Source", options=["All"] + list(sources)
            )
        with col3:
            label_filter = st.selectbox(
                "Select Innovation Type", options=["All"] + list(labels)
            )

        filtered_data = claims_data
        if tech_filter != "All":
            filtered_data = filtered_data[filtered_data["Tech"] == tech_filter]
        if source_filter != "All":
            filtered_data = filtered_data[
                filtered_data["Patent Source"] == source_filter
            ]
        if label_filter != "All":
            filtered_data = filtered_data[filtered_data["Labels"] == label_filter]

        if len(filtered_data) > 10000:
            filtered_data = sample_data(filtered_data, n=10000)

        text = " ".join(filtered_data["Text"])
        wordcloud = generate_wordcloud(text, mask, max_words=200)

        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

    # Bar Chart
    with st.container(border=True):
        st.header(":bar_chart: :rainbow[Product/Process Innovation Ratio]")
        st.markdown(
            "Compare the ratio of product and process innovations across different technologies."
        )

        source_filter_bar = st.selectbox(
            "Select Patent Source for Bar Chart", options=["All"] + list(sources)
        )
        sort_order = st.radio("Sort Order", options=["Ascending", "Descending"])

        filtered_bar_data = claims_data
        if source_filter_bar != "All":
            filtered_bar_data = filtered_bar_data[
                filtered_bar_data["Patent Source"] == source_filter_bar
            ]

        bar_data = (
            filtered_bar_data.groupby(["Tech", "Labels"])
            .size()
            .unstack()
            .fillna(0)
            .reset_index()
        )
        bar_data["Product/Process Ratio"] = (
            bar_data["Product Innovation"] / bar_data["Process Innovation"]
        )
        bar_data = bar_data.sort_values(
            by="Product/Process Ratio", ascending=(sort_order == "Ascending")
        )

        fig = plot_bar_chart(bar_data)
        st.plotly_chart(fig)

    # Sample Data and Summary Statistics
    st.header("Data Overview")
    st.markdown(
        "Explore a random sample of the dataset and view key summary statistics."
    )

    tab1, tab2 = st.tabs(["Summary Statistics", "Sample Data"])

    with tab1:
        display_summary_statistics(claims_data)

    with tab2:
        sampled_data = sample_data(claims_data, n=500)
        st.write(sampled_data)


if __name__ == "__main__":
    main()
