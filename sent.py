# ============================================
# NLP SENTIMENT ANALYSIS DASHBOARD
# ============================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import joblib

# ============================================
# Load trained ML model
# ============================================
@st.cache_resource
def load_model():
    return joblib.load("sentiment_tfidf_pipeline.pkl")

model = load_model()

# ============================================
# Load cleaned dataset
# ============================================
@st.cache_data
def load_data():
    df = pd.read_csv(r"G:\Python\sqlquery\miniproject\cleaned_reviews.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["review_length"] = df["review"].astype(str).apply(len)
    return df

df = load_data()

# ============================================
# Page setup
# ============================================
st.set_page_config(page_title="NLP Sentiment Analysis Dashboard", layout="wide")

st.title("📊 NLP Sentiment Analysis Dashboard")
st.caption("EDA insights and real-time sentiment prediction")

# ============================================
# 🔘 SIDEBAR MODE SELECTION
# ============================================
st.sidebar.title("📌 Navigation")

mode = st.sidebar.selectbox(
    "Choose Mode",
    ["📊 Exploratory Data Analysis (EDA)", "🧠 Sentiment Prediction"]
)

# =========================================================
# 🧠 SENTIMENT PREDICTION MODE
# =========================================================
if mode == "🧠 Sentiment Prediction":

    st.subheader("🧠 Real-Time Sentiment Prediction")

    user_input = st.text_area(
        "Enter review text:",
        placeholder="Type a review here...",
        height=120
    )

    if st.button("Predict Sentiment"):

        if user_input.strip() == "":
            st.error("Please enter some text")
        else:
            prediction = model.predict([user_input])[0]
            probs = model.predict_proba([user_input])[0]
            confidence = probs.max()

            st.success(f"🏷️ **Predicted Sentiment:** {prediction}")

            if confidence < 0.55:
                st.warning("⚠️ Prediction confidence is low. Model is uncertain.")

# =========================================================
# 📊 EDA MODE
# =========================================================
else:

    st.sidebar.subheader("📊 EDA Menu")

    question = st.sidebar.selectbox(
        "Select Analysis",
        [
            "1. Overall Sentiment Distribution",
            "2. Sentiment vs Rating",
            "3. Keywords per Sentiment",
            "4. Sentiment Trend Over Time",
            "5. Verified vs Non-Verified Users",
            "6. Review Length vs Sentiment",
            "7. Sentiment by Location",
            "8. Sentiment by Platform",
            "9. Sentiment by ChatGPT Version",
            "10. Negative Feedback Themes"
        ]
    )

    # ============================
    # 1. Overall Sentiment
    # ============================
    if question.startswith("1"):
        st.subheader("Overall Sentiment Distribution")
        st.bar_chart(df["sentiment"].value_counts())

    # ============================
    # 2. Sentiment vs Rating
    # ============================
    elif question.startswith("2"):
        crosstab = pd.crosstab(df["rating"], df["sentiment"], normalize="index") * 100
        st.dataframe(crosstab)
        st.bar_chart(crosstab)

    # ============================
    # 3. Keywords per Sentiment
    # ============================
    elif question.startswith("3"):
        sentiment_choice = st.selectbox(
            "Choose Sentiment",
            df["sentiment"].unique()
        )
        text = " ".join(df[df["sentiment"] == sentiment_choice]["review"].astype(str))
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        st.image(wc.to_array())

    # ============================
    # 4. Sentiment Trend
    # ============================
    elif question.startswith("4"):
        trend = df.groupby(
            [pd.Grouper(key="date", freq="M"), "sentiment"]
        ).size().unstack(fill_value=0)
        st.line_chart(trend)

    # ============================
    # 5. Verified Users
    # ============================
    elif question.startswith("5"):
        verified = pd.crosstab(
            df["verified_purchase"], df["sentiment"], normalize="index"
        ) * 100
        st.dataframe(verified)
        st.bar_chart(verified)

    # ============================
    # 6. Review Length
    # ============================
    elif question.startswith("6"):
        fig, ax = plt.subplots()
        sns.boxplot(x="sentiment", y="review_length", data=df, ax=ax)
        st.pyplot(fig)

    # ============================
    # 7. Location
    # ============================
    elif question.startswith("7"):
        loc = pd.crosstab(df["location"], df["sentiment"], normalize="index") * 100
        st.dataframe(loc)
        st.bar_chart(loc)

    # ============================
    # 8. Platform
    # ============================
    elif question.startswith("8"):
        plat = pd.crosstab(df["platform"], df["sentiment"], normalize="index") * 100
        st.dataframe(plat)
        st.bar_chart(plat)

    # ============================
    # 9. Version
    # ============================
    elif question.startswith("9"):
        ver = pd.crosstab(df["version"], df["sentiment"], normalize="index") * 100
        st.dataframe(ver)
        st.bar_chart(ver)

    # ============================
    # 10. Negative Themes
    # ============================
    elif question.startswith("10"):
        negative_text = " ".join(
            df[df["sentiment"] == "Negative"]["review"].astype(str)
        )
        wc = WordCloud(width=800, height=400, background_color="white").generate(negative_text)
        st.image(wc.to_array())
