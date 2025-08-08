# 🔍 RAGVision: AI-Powered Project Retrieval & Generation System

**RAGVision** is a smart academic project helper that retrieves relevant past projects using semantic search and suggests new project ideas with the help of **Retrieval-Augmented Generation (RAG)**. Built using **DeepSeek API**, **FAISS**, and **Streamlit**, it ensures your project ideas are both innovative and non-duplicative.

---

## 🌟 Key Objectives

- 🔍 **Smart Retrieval**: Fetch relevant past projects using semantic similarity.
- 🤖 **AI-Based Generation**: Suggest novel, personalized ideas via **RAG + DeepSeek**.
- 🧠 **Originality Check**: Uses cosine similarity to detect near-duplicate ideas.
- 📊 **Feasibility Insights**: Helps users assess tech stack, innovation level, and scope.
- 🚀 **Project Streamlining**: Simplifies academic/research project selection.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **DeepSeek API** | LLM-powered idea generation |
| **FAISS** | Fast semantic search and vector retrieval |
| **Streamlit** | UI for user interaction |
| **Cosine Similarity** | Idea duplication detection |
| **Jupyter Notebook** | Data processing & development |

---

## 📁 Folder Structure

```
ragvision/
├── app3.py                   # Streamlit app interface
├── RAGVision.ipynb           # Jupyter exploration notebook
├── requirements.txt          # Dependency list
├── README.md                 # Project documentation
├── .gitignore                # Files to ignore in repo
├── /data/                    # Contains your dataset
│   └── project_dataset.csv

```

---

## 🔐 API Key Setup (DeepSeek)

Create a file named `deepseek_key.txt` and paste your API key in it:

```
sk-your_deepseek_key_here
```

**Important:** Do not share this file publicly — it should be listed in `.gitignore`.

In `app3.py`, load it like this:
```python
with open("deepseek_key.txt", "r") as file:
    DEEPSEEK_API_KEY = file.read().strip()
```

---

## 📦 Installation

1. Clone the repo:
```bash
git clone https://github.com/yourusername/ragvision.git
cd ragvision
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app3.py
```

---

## 📊 Dataset

Your dataset (CSV or JSON) should be stored in the `data/` folder.

If it’s large or sensitive, consider adding it to `.gitignore`.

---

## 📓 Notebooks

Use `RAGVision.ipynb` to:
- Preprocess datasets
- Visualize embeddings
- Test FAISS retrieval
- Experiment with prompt engineering

---

---

## 📄 License

MIT License – feel free to use and contribute.

---

## 🤝 Contributions

Pull requests are welcome! For major changes, open an issue first.

---
