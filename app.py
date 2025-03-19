import streamlit as st
import json
import pandas as pd
from utils import get_article_links, analyze_articles, extract_topic_name

st.set_page_config(page_title="BBC News Summarizer", layout="centered")
st.title("📰 BBC News Summarizer & Sentiment Analyzer")

query = st.text_input("Enter a BBC topic URL or keyword:")
lines = st.number_input("Number of lines per summary (recommended: 3)", min_value=1, value=3)
start_page = st.number_input("Start page number", min_value=1, value=1)
end_page = st.number_input("End page number", min_value=start_page, value=3)

if st.button("Analyze"):
    if query:
        with st.spinner("Fetching and analyzing articles..."):
            company_name = extract_topic_name(query)
            article_links = get_article_links(query, int(start_page), int(end_page), min_articles=10)
            if article_links:
                analyze_articles(article_links, company_name, lines)
                st.success("Analysis complete! Displaying results below...")

                # Load generated JSON
                json_file = f"{company_name.lower()}_summary.json"
                with open(json_file, "r", encoding="utf-8") as f:
                    result = json.load(f)

                # Display Overall Conclusion
                st.subheader("📈 Overall Sentiment Conclusion")
                st.markdown(f"**{result['Overall Sentiment Conclusion']}**")

                # Sentiment Distribution
                st.subheader("📊 Sentiment Distribution")
                sentiment_dist = result["Comparative Sentiment Insights"]["Sentiment Distribution"]
                st.bar_chart(pd.Series(sentiment_dist))

                # Common Topics
                st.subheader("📚 Common Topics Across All Articles")
                common_topics = result["Comparative Sentiment Insights"]["Topic Analysis"]["Common Topics"]
                st.write(", ".join(common_topics) if common_topics else "No common topics found.")

                # Comparisons
                st.subheader("🔎 Article Comparisons")
                for c in result["Comparative Sentiment Insights"]["Comparisons"]:
                    st.write(f"- **{c['Comparison']}** ➡️ {c['Impact']}")

                # Article Details
                st.subheader("📝 Article Summaries")
                articles_df = pd.DataFrame(result["Articles"])
                st.dataframe(articles_df)

                # Download CSV
                csv_data = articles_df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Articles CSV", data=csv_data, file_name=f"{company_name.lower()}_articles.csv", mime="text/csv")

                # Audio player for Hindi sentiment summary
                st.subheader("🔊 Hindi Sentiment Summary")
                audio_file = f"{company_name.lower()}_sentiment_hindi.mp3"
                st.audio(audio_file)

            else:
                st.warning("No articles found for this query.")
    else:
        st.error("Please enter a keyword or URL.")
