import os
import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

# Initialize environment variables
load_dotenv()

# Initialize Groq LLM
def get_llm():
    return ChatGroq(
        model="llama3-70b-8192",
        temperature=0.2,
        max_tokens=512,
        timeout=60,
        max_retries=3,
        groq_api_key=st.secrets["GROQ_API_KEY"],
    )

# Setup vector database from PDF
@st.cache_resource(show_spinner="Processing document...")
def setup_vector_db(uploaded_file):
    with st.spinner("Loading and chunking document..."):
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=128,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
    
    with st.spinner("Creating knowledge base..."):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        return FAISS.from_documents(chunks, embeddings)

# Generate answer with conversation memory
def generate_answer(vector_db, query):
    llm = get_llm()
    
    # Enhanced prompt with conversation history
    prompt_template = """You are an expert on the document content. Use the following context and conversation history to answer the question:
    
    Conversation History:
    {chat_history}

    Document Context:
    {context}

    Current Question: {question}

    Guidelines:
    1. Answer concisely (max 3 sentences)
    2. Maintain context from previous questions
    3. If information isn't in document, say "I don't have that information"
    4. For NEPA-related questions, reference specific sections when possible
    5. Use natural language that flows in conversation
    
    Helpful Answer:"""
    
    QA_PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question", "chat_history"]
    )
    
    # Create conversation chain with memory
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 15}
        ),
        memory=st.session_state.memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=False,
        get_chat_history=lambda h: h
    )
    
    result = qa_chain({"question": query})
    return result['answer']

# Main processing function
def process_query(query, vector_db):
    st.info(f"**Processing query:** {query}")
    
    # Generate answer from document with memory
    with st.spinner("Analyzing document and conversation history..."):
        answer = generate_answer(vector_db, query)
    
    return answer

# Streamlit UI
def main():
    st.set_page_config(page_title="Document Expert System", layout="wide")
    st.title("📄 Document Expert System with Memory")
    st.caption("Upload a PDF document and have a conversation about its content")
    
    # Initialize session state components
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = None
    
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(
            k=5,  # Remember last 5 exchanges
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add initial assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hi! I'm your document expert. Upload a PDF and ask me anything about its content."
        })
    
    # Sidebar controls
    with st.sidebar:
        st.subheader("Document Management")
        uploaded_file = st.file_uploader("Upload knowledge document (PDF)", type="pdf")
        
        st.divider()
        st.subheader("Conversation Memory")
        st.info(f"Remembering last {st.session_state.memory.k} exchanges")
        
        if st.button("Clear Conversation History"):
            st.session_state.memory.clear()
            st.session_state.messages = [
                {"role": "assistant", "content": "Conversation history cleared. How can I help?"}
            ]
            st.success("History cleared!")
    
    # Process document when uploaded
    if uploaded_file and st.session_state.vector_db is None:
        st.session_state.vector_db = setup_vector_db(uploaded_file)
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"✅ Document processed! Ask me anything about its content."
        })
        st.rerun()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Query input
    query = st.chat_input("Ask about the document content:")
    
    # Process query when entered
    if query:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)
        
        # Process query
        if st.session_state.vector_db:
            with st.chat_message("assistant"):
                answer = process_query(query, st.session_state.vector_db)
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            st.warning("Please upload a PDF document first")

if __name__ == "__main__":
    main()