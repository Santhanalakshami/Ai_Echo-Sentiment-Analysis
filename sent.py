# sentiment_dashboard.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

# ============================
# Load Data
# ============================
@st.cache_data
def load_data():
    df = pd.read_csv(r"G:\Python\sqlquery\miniproject\cleaned_reviews.csv")
    # Expect columns like: ['review','sentiment','rating','verified_purchase',
    #                       'date','location','platform','version']
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['review_length'] = df['review'].astype(str).apply(len)
    return df

df = load_data()

st.title("📊 Sentiment Analysis Dashboard")
st.write("Interactive insights from user reviews")

# Sidebar
question = st.sidebar.selectbox("Select a Question", [
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
])

# ============================
# 1. Overall Sentiment Distribution
# ============================
if question.startswith("1"):
    st.subheader("Overall Sentiment of User Reviews")
    sentiment_counts = df['sentiment'].value_counts(normalize=True) * 100
    st.bar_chart(sentiment_counts)

# ============================
# 2. Sentiment vs Rating
# ============================
elif question.startswith("2"):
    st.subheader("How Does Sentiment Vary by Rating?")
    crosstab = pd.crosstab(df['rating'], df['sentiment'], normalize="index") * 100
    st.write(crosstab)
    st.bar_chart(crosstab)

# ============================
# 3. Keywords per Sentiment
# ============================
elif question.startswith("3"):
    st.subheader("Keywords Associated with Each Sentiment")
    sentiment_choice = st.selectbox("Choose Sentiment", df['sentiment'].unique())
    text = " ".join(df[df['sentiment']==sentiment_choice]['review'].astype(str))
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    st.image(wc.to_array())

# ============================
# 4. Sentiment Trend Over Time
# ============================
elif question.startswith("4"):
    st.subheader("Sentiment Trends Over Time")
    trend = df.groupby([pd.Grouper(key="date", freq="M"), "sentiment"]).size().unstack(fill_value=0)
    st.line_chart(trend)

# ============================
# 5. Verified vs Non-Verified
# ============================
elif question.startswith("5"):
    st.subheader("Do Verified Users Leave Different Reviews?")
    crosstab = pd.crosstab(df['verified_purchase'], df['sentiment'], normalize="index") * 100
    st.write(crosstab)
    st.bar_chart(crosstab)

# ============================
# 6. Review Length vs Sentiment
# ============================
elif question.startswith("6"):
    st.subheader("Are Longer Reviews More Positive or Negative?")
    length_stats = df.groupby("sentiment")['review_length'].mean()
    st.write(length_stats)
    sns.boxplot(x="sentiment", y="review_length", data=df)
    st.pyplot(plt.gcf())

# ============================
# 7. Sentiment by Location
# ============================
elif question.startswith("7"):
    st.subheader("Which Locations Show Most Positive/Negative Sentiment?")
    loc_counts = pd.crosstab(df['location'], df['sentiment'], normalize="index") * 100
    st.write(loc_counts)
    st.bar_chart(loc_counts)

# ============================
# 8. Sentiment by Platform
# ============================
elif question.startswith("8"):
    st.subheader("Sentiment by Platform (Web vs Mobile)")
    plat_counts = pd.crosstab(df['platform'], df['sentiment'], normalize="index") * 100
    st.write(plat_counts)
    st.bar_chart(plat_counts)

# ============================
# 9. Sentiment by Version
# ============================
elif question.startswith("9"):
    st.subheader("Sentiment Across ChatGPT Versions")
    ver_counts = pd.crosstab(df['version'], df['sentiment'], normalize="index") * 100
    st.write(ver_counts)
    st.bar_chart(ver_counts)

# ============================
# 10. Negative Feedback Themes
# ============================
elif question.startswith("10"):
    st.subheader("Common Themes in Negative Reviews")
    negative_text = " ".join(df[df['sentiment']=="Negative"]['review'].astype(str))
    wc = WordCloud(width=800, height=400, background_color="white").generate(negative_text)
    st.image(wc.to_array())
    st.write("Word Cloud shows the recurring pain points in negative feedback.")
