import streamlit as st
import os
from langchain.embeddings import OpenAIEmbeddings  # or use other available embedding models
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import ChatPromptTemplate
import tempfile
from pydantic import ValidationError  # Import ValidationError from Pydantic

# Configure the Google API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCwzEFcyhmlFNLukx8sH6jruQwhHk25js8"
# Set the OpenAI API key
#os.environ["OPENAI_API_KEY"] = "sk-proj-JpA7BaO_7OyJs9YA0ZVAcI_fmcc605fax5d-rMCy0O_E64TltmoeV45eN4mG8djTxGpGMmcH-T3BlbkFJg4-CX9UTwMKQECcg1wSmfJZ4FqoM33lEfQ-61fhpvVCcEqITagwzVeXZOl2B_yXErbpNNJZIA"  # Add your OpenAI API key here

# Initialize the language model
try:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.0)
except NameError as e:
    st.error(f"Failed to initialize LLM: {e}")
    llm = None  # Set llm to None if initialization fails

# Define the question prompt
question = '''Please analyze the following documents, which may contain multiple languages, and generate a structured survey paper format covering all relevant details from each PDF...'''

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
        embeddings = OpenAIEmbeddings()  # Initialize embeddings
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
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"document_variable_name": "context"}
    )
    return rag_chain

def load_and_split_documents(pdf_files):
    documents = []
    for pdf_file in pdf_files:
        # Save the uploaded file to a temporary file
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
    answer = rag_chain.invoke({"query": question})
    return answer['result']

# Streamlit app layout
st.title("AI-Powered PDF Processor")

# Sidebar for option selection
option = st.sidebar.selectbox("Choose an Option:", ["AI-Powered PDF Summarizer", "Chat with Multiple PDFs"])

if option == "AI-Powered PDF Summarizer":
    st.header("AI-Powered PDF Summarizer")
    uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type="pdf")
    
    if st.button("Generate Summary"):
        if uploaded_files:
            pdf_files = [pdf_file for pdf_file in uploaded_files]  # Corrected to get the file objects
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
            pdf_files = [pdf_file for pdf_file in uploaded_files]  # Corrected to get the file objects
            documents = load_and_split_documents(pdf_files)
            retriever = create_retriever(documents)
            if retriever is None:
                st.error("Failed to create retriever.")
            else:
                rag_chain = create_rag_chain(retriever)

                chat_input = st.text_input("Ask a question about the documents:")
                if chat_input:
                    answer = rag_chain.invoke({"query": chat_input})
                    st.write(answer['result'])
        else:
            st.error("Please upload PDF files for chatting.")
