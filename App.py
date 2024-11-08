import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain, StuffDocumentsChain, RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document

# Configure the Google API key
os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"

# Initialize the language model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.0)

# Define the question prompt
question = '''Please analyze the following documents, which may contain multiple languages, and generate a structured survey paper format covering all relevant details from each PDF...'''

# Define prompt template
system_prompt = (
    """Your instructions for generating the survey paper structure as per user requirements."""
    "\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])

# Initialize LLM chain
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Define RAG chain
def create_retriever(documents):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    faiss_index = FAISS.from_documents(documents, embeddings)
    retriever = faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": 20})
    return retriever

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
        loader = PyPDFLoader(pdf_file)
        doc = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = text_splitter.split_documents(doc)
        
        for i, doc_chunk in enumerate(split_docs):
            doc_chunk.metadata['source'] = pdf_file
            doc_chunk.metadata['page'] = i
        documents.extend(split_docs)
    return documents

def qa_system(question, pdf_files):
    documents = load_and_split_documents(pdf_files)
    retriever = create_retriever(documents)
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
            pdf_files = [pdf_file.name for pdf_file in uploaded_files]
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
            pdf_files = [pdf_file.name for pdf_file in uploaded_files]
            documents = load_and_split_documents(pdf_files)
            retriever = create_retriever(documents)
            rag_chain = create_rag_chain(retriever)

            chat_input = st.text_input("Ask a question about the documents:")
            if chat_input:
                answer = rag_chain.invoke({"query": chat_input})
                st.write(answer['result'])
        else:
            st.error("Please upload PDF files for chatting.")
