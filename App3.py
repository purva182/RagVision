import streamlit as st
import os
import faiss
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from fpdf import FPDF

st.set_page_config(page_title="RAGVision: AI-Powered Project Retrieval and Ideation System ", layout="wide")

st.title("🚀 RAGVision: AI-Powered Project Retrieval and Ideation System")

@st.cache_resource(show_spinner=False)
def init_resources():
    client = OpenAI(
        api_key= "sk-7840f7614f794673bb09d0c845f6b68a",
        base_url="https://api.deepseek.com"
    )
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return client, embedding_model

@st.cache_data(show_spinner="Loading project dataset and index...")
def load_data():
    df = pd.read_csv("Final_Project_Dataset_with_400_New_Projects.csv")
    if 'Feasibility Score' not in df.columns:
        df['Feasibility Score'] = np.random.randint(6, 10, size=len(df))
    faiss_index = faiss.read_index("faiss_index.idx")
    return df, faiss_index

client, embedding_model = init_resources()
df, faiss_index = load_data()

# --- Graphs ---
def show_feasibility_chart():
    fig, ax = plt.subplots()
    df['Feasibility Score'].value_counts().sort_index().plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title("Distribution of Feasibility Scores")
    ax.set_xlabel("Feasibility Score")
    ax.set_ylabel("Number of Projects")
    st.pyplot(fig)

def show_tech_chart():
    tech_counts = df['Technologies Used'].str.split(', ').explode().value_counts().nlargest(10)
    fig = px.pie(values=tech_counts.values, names=tech_counts.index, title="Top 10 Technologies Used")
    st.plotly_chart(fig)

def show_domain_distribution():
    if 'Domain' in df.columns:
        domain_counts = df['Domain'].value_counts()
        fig = px.bar(x=domain_counts.index, y=domain_counts.values,
                     labels={'x': 'Domain', 'y': 'Number of Projects'},
                     title="Projects by Domain")
        st.plotly_chart(fig)

def show_title_keyword_frequency():
    all_words = " ".join(df['Project Name'].dropna()).lower().split()
    keywords = pd.Series(all_words).value_counts().head(10)
    fig, ax = plt.subplots()
    keywords.plot(kind='bar', color='orange', ax=ax)
    ax.set_title("Top 10 Keywords in Project Titles")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

def show_avg_feasibility_per_tech():
    exploded_df = df.dropna(subset=['Technologies Used']).copy()
    exploded_df['Technologies Used'] = exploded_df['Technologies Used'].str.split(', ')
    exploded_df = exploded_df.explode('Technologies Used')
    avg_score = exploded_df.groupby('Technologies Used')['Feasibility Score'].mean().sort_values(ascending=False).head(10)
    fig = px.bar(x=avg_score.index, y=avg_score.values,
                 labels={'x': 'Technology', 'y': 'Avg. Feasibility Score'},
                 title="Avg. Feasibility Score per Technology")
    st.plotly_chart(fig)

# --- Main Functionality ---
def retrieve_context(user_query, top_k=3):
    query_emb = embedding_model.encode([user_query]).astype('float32')
    distances, indices = faiss_index.search(query_emb, top_k)
    if indices.size == 0 or np.all(indices == -1):
        return "No similar projects found."
    return "\n\n".join(
        f"### {df.iloc[idx]['Project Name']}\n"
        f"**Description**: {df.iloc[idx]['Project Description']}\n\n"
        f"**Technologies**: {df.iloc[idx]['Technologies Used']}\n"
        f"**Feasibility**: {df.iloc[idx]['Feasibility Score']}/10"
        f"**Accuracy**: {df.iloc[idx]['Accuracy']}"
        for idx in indices[0] if idx < len(df)
    )

def generate_idea(user_query, retrieved_context, skill_level):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a creative project ideation assistant."},
            {"role": "user", "content": f"""
                Generate a NEW project idea based on:
                USER INTEREST: {user_query}
                RELATED PROJECTS: {retrieved_context}
                SKILL LEVEL: {skill_level}

                FORMAT:
                1. **Title**: Catchy name
                2. **Concept**: 2-3 sentence overview
                3. **Innovation**: Unique aspects
                4. **Tech Stack**: Recommended technologies
                5. **Feasibility**: Difficulty (1-10)
            """}
        ],
        temperature=0.7,
        max_tokens=800
    )
    return response.choices[0].message.content.strip()

# --- UI ---
with st.sidebar:
    st.header("📊 Data Visualizations")
    if st.checkbox("Show Feasibility Score Distribution"):
        show_feasibility_chart()
    if st.checkbox("Show Top 10 Technologies Used"):
        show_tech_chart()
    if st.checkbox("Show Keywords in Titles"):
        show_title_keyword_frequency()
    if st.checkbox("Show Avg. Feasibility per Technology"):
        show_avg_feasibility_per_tech()

with st.form("idea_form"):
    query = st.text_area("Describe your project interest:", value="AI for education")
    skill_level = st.selectbox("Select your current skill level:", ["Beginner", "Intermediate", "Advanced"])
    submitted = st.form_submit_button("Generate Idea")

if submitted:
    with st.spinner("🔎 Finding similar projects..."):
        context = retrieve_context(query)
    st.subheader("📌 Similar Projects")
    st.markdown(context)
    with st.spinner("⚙️ Generating innovative idea..."):
        idea = generate_idea(query, context, skill_level)
    st.subheader("💡 Your Custom Project Idea")
    st.markdown(idea)

    # Export options
    st.download_button("📥 Download Idea (Text)", data=idea, file_name=f"project_idea_{query[:20].replace(' ', '_')}.txt")
    def export_to_pdf(title, content):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, f"Project Title: {title}\n\n{content}")
        pdf_path = f"project_idea_{title[:20].replace(' ', '_')}.pdf"
        pdf.output(pdf_path)
        return pdf_path

    if st.button("📄 Export as PDF"):
        pdf_path = export_to_pdf(query, idea)
        with open(pdf_path, "rb") as f:
            st.download_button("📥 Download PDF", f, file_name=os.path.basename(pdf_path))
