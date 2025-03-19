import streamlit as st
import json
import pandas as pd
import requests
from pathlib import Path

# Streamlit page setup
st.set_page_config(page_title="BBC News Summarizer", layout="centered")
st.title("📰 BBC News Summarizer & Sentiment Analyzer")

# Input fields
query = st.text_input("Enter a BBC topic URL or keyword:")
lines = st.number_input("Number of lines per summary (recommended: 3)", min_value=1, value=3)
start_page = st.number_input("Start page number", min_value=1, value=1)
end_page = st.number_input("End page number", min_value=start_page, value=3)

if "result" not in st.session_state:
    st.session_state["result"] = None
if "company_name" not in st.session_state:
    st.session_state["company_name"] = None

# Analyze button logic via API
if st.button("Analyze"):
    if query:
        with st.spinner("Fetching and analyzing articles from backend..."):
            response = requests.post("http://localhost:8000/analyze/", json={
                "query": query,
                "start_page": int(start_page),
                "end_page": int(end_page),
                "lines": int(lines)
            })
            data = response.json()
            if data["success"]:
                st.session_state["result"] = data["result"]
                st.session_state["company_name"] = data["company_name"]
                st.success("Analysis complete! Scroll down to see results.")
            else:
                st.error(data["message"])
    else:
        st.error("Please enter a BBC topic URL or keyword.")

# Show results and compare articles via API
if st.session_state["result"]:
    result = st.session_state["result"]
    company_name = st.session_state["company_name"]

    st.subheader("📈 Overall Sentiment Conclusion")
    st.markdown(f"**{result['Overall Sentiment Conclusion']}**")

    st.subheader("📊 Sentiment Distribution")
    sentiment_dist = result["Comparative Sentiment Insights"]["Sentiment Distribution"]
    st.bar_chart(pd.Series(sentiment_dist))

    st.subheader("📚 Common Topics Across All Articles")
    common_topics = result["Comparative Sentiment Insights"]["Topic Analysis"]["Common Topics"]
    st.write(", ".join(common_topics) if common_topics else "No common topics found.")

    st.subheader("🔎 Auto Comparisons from Analysis")
    for c in result["Comparative Sentiment Insights"]["Comparisons"]:
        st.write(f"- **{c['Comparison']}** ➡️ {c['Impact']}")

    st.subheader("📝 Article Summaries")
    articles_df = pd.DataFrame(result["Articles"])
    st.dataframe(articles_df)

    st.subheader("🔎 Compare Two Articles Manually")
    if len(result["Articles"]) >= 2:
        index1 = st.number_input("Select first article index", min_value=0, max_value=len(result["Articles"]) - 1, key="index1")
        index2 = st.number_input("Select second article index", min_value=0, max_value=len(result["Articles"]) - 1, key="index2")

        if st.button("Compare Selected Articles"):
            compare_resp = requests.post("http://localhost:8000/compare/", json={
                "articles": result["Articles"],
                "index1": index1,
                "index2": index2
            })
            compare_data = compare_resp.json()
            st.markdown(f"**Article 1:** {compare_data['Article 1']}")
            st.markdown(f"**Sentiment:** {compare_data['Article 1 Sentiment']}")
            st.markdown(f"**Article 2:** {compare_data['Article 2']}")
            st.markdown(f"**Sentiment:** {compare_data['Article 2 Sentiment']}")
            st.info(compare_data["Comparison Summary"])
    else:
        st.warning("Not enough articles to compare.")

    csv_data = articles_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Articles CSV", data=csv_data, file_name=f"{company_name.lower()}_articles.csv", mime="text/csv")

    st.subheader("🔊 Hindi Sentiment Summary")
    audio_file = Path(f"results/{company_name.lower()}_sentiment_hindi.mp3")
    if audio_file.exists():
        st.audio(str(audio_file))
    else:
        st.warning("Hindi sentiment audio not found.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit and FastAPI.")

