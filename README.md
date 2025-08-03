# Smart Contextual Question Answering with Local + Web Knowledge

This project is an intelligent question-answering system that combines **local PDF knowledge** with **live web search and scraping** to provide accurate, up-to-date answers.

It is designed to:
- Extract contextual knowledge from a PDF (e.g., travel brochures, reports)
- Decide if the local context is sufficient to answer a user's question
- If not, search and scrape the web using specialized agents
- Generate a final, human-like response using powerful LLMs like **LLaMA 3 (70B)** and **Gemini Flash**

## Features
- 📄 **PDF Parsing**: Parses and chunks PDFs using `langchain` and FAISS for semantic search  
- 🤖 **LLM Reasoning**: Uses LLaMA 3 and Gemini to decide whether local documents are sufficient  
- 🌐 **Web Agents**: Leverages `CrewAI` agents for:
  - Google-style search via `SerperDevTool`
  - Web scraping via `ScrapeWebsiteTool`
- 📚 **Hybrid Knowledge Flow**:
  - Uses local documents *if possible*
  - Falls back to web scraping if needed
- Fully functional end-to-end pipeline with a working example on *Chitwan National Park travel info*

---
## Technologies Used

- `langchain`, `FAISS`, `ChatGroq`, `CrewAI`
- LLMs: **LLaMA3-70B** (via Groq) & **Gemini Flash**
- Embeddings: `sentence-transformers/all-mpnet-base-v2`
- Web search: `SerperDevTool`  
- Web scraping: `ScrapeWebsiteTool`
- Environment variables managed via `.env`
- 
## How to Run

1. Clone the repository and install requirements:
   ```bash
   pip install -r requirements.txt
