import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import unicodedata

# Streamlit Page Config
st.set_page_config(page_title="SmartDocs 💬", page_icon="📄", layout="wide")

# Load Environment Variables
load_dotenv()
genai_api_key = os.getenv("GOOGLE_API_KEY")

# Helper to clean text
def clean_text(text):
    """
    Cleans the text by normalizing and replacing invalid characters.
    """
    return unicodedata.normalize("NFKD", text).encode("utf-8", "ignore").decode("utf-8")

# Extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

# Split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return [chunk.strip() for chunk in text_splitter.split_text(text) if chunk.strip()]

# Vector store for QA
def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in 
    the provided context, just say, "Answer is not available in the context."\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Process user input
def process_user_input(user_question, pdf_docs):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

        docs = vector_store.similarity_search(user_question, k=3)

        if not docs:
            return "No relevant information found in the context to answer your question."

        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

        return response["output_text"]
    except Exception as e:
        st.error(f"An error occurred while processing your query: {e}")
        return "An error occurred. Please try again."

# Export chat history
def export_chat_history(chat_history, file_format):
    try:
        if not chat_history:
            st.warning("No chat history available to export.")
            return

        cleaned_history = "\n\n".join(
            [
                f"Question: {clean_text(entry['query'])}\nAnswer: {clean_text(entry['response'])}"
                for entry in chat_history
            ]
        )

        if file_format == "txt":
            st.download_button(
                label="Download Chat History as TXT",
                data=cleaned_history,
                file_name="chat_history.txt",
                mime="text/plain",
            )
        elif file_format == "pdf":
            from fpdf import FPDF

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)

            for line in cleaned_history.split("\n"):
                pdf.cell(200, 10, txt=line, ln=True)

            pdf_output = pdf.output(dest="S").encode("latin1")

            st.download_button(
                label="Download Chat History as PDF",
                data=pdf_output,
                file_name="chat_history.pdf",
                mime="application/pdf",
            )
    except Exception as e:
        st.error(f"An error occurred during export: {e}")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

# Main Function
def main():
    st.title("SmartDocs 💬")
    st.markdown("Upload your PDFs, process them, and ask questions using AI.")

    with st.sidebar:
        st.subheader("📂 Upload PDFs")
        pdf_docs = st.file_uploader("Upload your PDFs", accept_multiple_files=True)

        if st.button("Process PDFs"):
            if not pdf_docs:
                st.error("Please upload at least one PDF before processing.")
            else:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.session_state.pdf_processed = True
                    st.success("PDFs processed successfully!")

    user_question = st.text_input("💡 Ask Your Question", placeholder="Type your question here...")
    submit_button = st.button("Submit")

    if submit_button:
        if not st.session_state.pdf_processed:
            st.warning("Please upload and process your PDFs before asking questions.")
        elif not user_question.strip():
            st.warning("Please enter a question.")
        else:
            response = process_user_input(user_question, pdf_docs)
            st.session_state.chat_history.append({"query": user_question, "response": response})

    with st.container():
        st.subheader("Chat History")
        if st.session_state.chat_history:
            for entry in st.session_state.chat_history:
                st.markdown(f"**You:** {entry['query']}")
                st.markdown(f"**AI:** {entry['response']}")
                st.markdown("---")

        st.subheader("Export Chat History")
        col1, col2 = st.columns(2)
        with col1:
            export_chat_history(st.session_state.chat_history, "txt")
        # with col2:
        #     export_chat_history(st.session_state.chat_history, "pdf")

if __name__ == "__main__":
    main()
