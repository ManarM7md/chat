import streamlit as st
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import ChatPromptTemplate
import tempfile
from pydantic import ValidationError

# Configure the Google API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCwzEFcyhmlFNLukx8sH6jruQwhHk25js8"
# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# Check for sentence_transformers package and choose embeddings
try:
    from langchain.embeddings import SentenceTransformerEmbeddings
    embedding_model = "sentence_transformers"  # Mark model as sentence_transformers
except ImportError:
    st.warning("sentence_transformers package not found. Using HuggingFaceEmbeddings instead.")
    from langchain.embeddings import HuggingFaceEmbeddings
    embedding_model = "huggingface"

try:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.0)
except NameError as e:
    st.error(f"Failed to initialize LLM: {e}")
    llm = None

# Define the question prompt
question = '''
    Please analyze the following multilingual PDF documents to generate a structured survey paper.
    The paper should be comprehensive and well-organized, capturing all relevant information from each PDF.
    [Content truncated for brevity]
'''

# Define prompt template
system_prompt = (
    """Your instructions for generating the survey paper structure as per user requirements."""
    "\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])

# Initialize LLM chain only if llm is successfully initialized
if llm:
    llm_chain = LLMChain(llm=llm, prompt=prompt)

def create_retriever(documents):
    try:
        if embedding_model == "sentence_transformers":
            embeddings = SentenceTransformerEmbeddings()
        else:
            embeddings = HuggingFaceEmbeddings()

        faiss_index = FAISS.from_documents(documents, embeddings)
        retriever = faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": 20})
        return retriever
    except ValidationError as e:
        st.error(f"Validation error while creating embeddings: {e}")
        return None
    except Exception as e:
        st.error(f"An error occurred while creating the retriever: {e}")
        return None

def create_rag_chain(retriever):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"document_variable_name": "context"}
    )

def load_and_split_documents(pdf_files):
    documents = []
    for pdf_file in pdf_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_file.read())
            temp_file_path = temp_file.name

        try:
            loader = PyPDFLoader(temp_file_path)
            doc = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            split_docs = text_splitter.split_documents(doc)
            
            for i, doc_chunk in enumerate(split_docs):
                doc_chunk.metadata['source'] = temp_file_path
                doc_chunk.metadata['page'] = i
            documents.extend(split_docs)
        except ValueError as e:
            st.error(f"Error loading file {pdf_file.name}: {e}")
    
    return documents

def qa_system(question, pdf_files):
    documents = load_and_split_documents(pdf_files)
    retriever = create_retriever(documents)
    if retriever is None:
        return "Retrieval failed due to previous errors."
    
    rag_chain = create_rag_chain(retriever)
    try:
        answer = rag_chain.invoke({"query": question})
        return answer['result']
    except Exception as e:
        st.error(f"Error during querying: {e}")
        return "Error occurred while processing the question."

# Streamlit app layout
st.title("AI-Powered PDF Processor")

# Sidebar for option selection
option = st.sidebar.selectbox("Choose an Option:", ["AI-Powered PDF Summarizer", "Chat with Multiple PDFs"])

if option == "AI-Powered PDF Summarizer":
    st.header("AI-Powered PDF Summarizer")
    uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type="pdf")
    
    if st.button("Generate Summary"):
        if uploaded_files:
            pdf_files = [pdf_file for pdf_file in uploaded_files]
            answer = qa_system(question, pdf_files)
            st.subheader("Generated Summary")
            st.write(answer)
        else:
            st.error("Please upload PDF files to summarize.")

elif option == "Chat with Multiple PDFs":
    st.header("Chat with Multiple PDFs")
    uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type="pdf")
    
    if st.button("Start Chat"):
        if uploaded_files:
            pdf_files = [pdf_file for pdf_file in uploaded_files]
            documents = load_and_split_documents(pdf_files)
            retriever = create_retriever(documents)
            if retriever is None:
                st.error("Failed to create retriever.")
            else:
                rag_chain = create_rag_chain(retriever)

                chat_input = st.text_input("Ask a question about the documents:")
                if chat_input:
                    try:
                        answer = rag_chain.invoke({"query": chat_input})
                        st.write(answer['result'])
                    except Exception as e:
                        st.error(f"Error during querying: {e}")
        else:
            st.error("Please upload PDF files for chatting.")
