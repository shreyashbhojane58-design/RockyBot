import os
import streamlit as st
import pickle
from dotenv import load_dotenv
from groq import Groq

# LangChain Imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


load_dotenv()

st.set_page_config(page_title="RockyBot Pro", layout="wide")
st.title("RockyBot Pro: Research Tool 📈 (URLs + PDFs Supported)")

# Initialize Groq Client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Sidebar
st.sidebar.title("Input Sources")

# 1. URL Input
st.sidebar.subheader("News Article URLs")
urls = []
for i in range(5):
    url = st.sidebar.text_input(f"URL {i+1}", key=f"url_{i}")
    if url:
        urls.append(url)

# 2. PDF Upload
st.sidebar.subheader("Upload PDF Files")
uploaded_pdfs = st.sidebar.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

process_clicked = st.sidebar.button("Process URLs & PDFs 🚀")
file_path = "faiss_store.pkl"
main_placeholder = st.empty()

# -------------------------
# GROQ LLM FUNCTION
# -------------------------
def groq_llm(prompt):
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Latest working model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error with LLM: {str(e)}"

# -------------------------
# PROCESS URLs + PDFs
# -------------------------
if process_clicked:
    if not urls and not uploaded_pdfs:
        st.error("Please provide at least one URL or upload one PDF.")
    else:
        docs = []
        main_placeholder.text("Processing your sources...")

        # Process URLs
        if urls:
            with st.spinner(f"Loading {len(urls)} URLs..."):
                try:
                    loader = UnstructuredURLLoader(urls=urls)
                    url_docs = loader.load()
                    for doc in url_docs:
                        doc.metadata["source"] = "URL"
                    docs.extend(url_docs)
                    st.success(f"Loaded {len(url_docs)} documents from URLs")
                except Exception as e:
                    st.error(f"URL loading failed: {e}")

        # Process PDFs
        if uploaded_pdfs:
            with st.spinner(f"Extracting text from {len(uploaded_pdfs)} PDFs..."):
                for uploaded_pdf in uploaded_pdfs:
                    # Save temporarily
                    temp_path = f"temp_{uploaded_pdf.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_pdf.getbuffer())

                    try:
                        loader = PyPDFLoader(temp_path)
                        pdf_pages = loader.load()
                        for i, page in enumerate(pdf_pages):
                            page.metadata["source"] = f"PDF: {uploaded_pdf.name}"
                            page.metadata["page"] = i + 1
                        docs.extend(pdf_pages)
                        os.remove(temp_path)  # Clean up
                    except Exception as e:
                        st.error(f"Failed to read {uploaded_pdf.name}: {e}")
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

                st.success(f"Extracted text from {len(uploaded_pdfs)} PDFs")

        if docs:
            with st.spinner("Splitting text into chunks..."):
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", " ", ""]
                )
                split_docs = text_splitter.split_documents(docs)

            with st.spinner("Creating vector database... This may take a minute ⏳"):
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vectorstore = FAISS.from_documents(split_docs, embeddings)

                with open(file_path, "wb") as f:
                    pickle.dump(vectorstore, f)

            main_placeholder.success(f"Ready! Processed {len(split_docs)} chunks from {len(urls)} URLs and {len(uploaded_pdfs)} PDFs")
        else:
            st.error("No content was loaded. Check your inputs.")

# -------------------------
# ASK QUESTIONS
# -------------------------
query = st.text_input("Ask a question about the uploaded PDFs or articles:", key="query")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

        # Retrieve relevant chunks
        relevant_docs = retriever.invoke(query)

        context = "\n\n".join([
            f"Source: {doc.metadata.get('source', 'Unknown')} | Page: {doc.metadata.get('page', 'N/A')}\nContent: {doc.page_content}"
            for doc in relevant_docs
        ])

        prompt = f"""
You are an expert research assistant. Answer the question using ONLY the provided context from PDFs and web articles.

Context:
{context}

Question: {query}

Answer clearly and cite sources like:
- [PDF: filename.pdf, Page X]
- [URL]

If you don't know, say "I don't have enough information."
"""

        with st.spinner("Thinking..."):
            answer = groq_llm(prompt)

        st.subheader("Answer")
        st.markdown(answer)

        st.subheader("Sources Used")
        for i, doc in enumerate(relevant_docs):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page")
            if "PDF:" in source:
                st.write(f"📄 {source}" + (f", Page {page}" if page else ""))
            else:
                st.write(f"🌐 {source}")
    else:

        st.warning("Please click 'Process URLs & PDFs' first!")

