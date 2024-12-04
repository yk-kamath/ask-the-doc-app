import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, WebBaseLoader
import tempfile
import pysqlite3
import sys

# Fix the sqlite3 module issue
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Page configuration (must be the first Streamlit command)
st.set_page_config(page_title='🦜🔗 Enhanced Ask the Doc App')

# Initialize persistent storage for document names
if 'document_list' not in st.session_state:
    st.session_state['document_list'] = []

def add_to_sidebar(doc_name):
    """Update the sidebar with the new document."""
    if doc_name not in st.session_state['document_list']:
        st.session_state['document_list'].append(doc_name)

def load_document(file=None, url=None):
    """Load a document from a file or URL."""
    if file is not None:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name
        
        # Load a PDF document from the temporary file
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        add_to_sidebar(file.name)
        return documents
    elif url is not None:
        # Load a document from a website
        loader = WebBaseLoader(url)
        documents = loader.load()
        add_to_sidebar(url)
        return documents

def generate_response(documents, openai_api_key, query_text):
    """Generate a response from the loaded documents."""
    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    # Select embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # Create a vectorstore from documents and use ChromaDB for persistence
    db = Chroma.from_documents(texts, embeddings, persist_directory="chromadb_storage")
    db.persist()  # Persist the embeddings to disk
    retriever = db.as_retriever()
    # Create QA chain
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=openai_api_key),
        chain_type='stuff',
        retriever=retriever
    )
    return qa.run(query_text)

# Sidebar for loaded documents
with st.sidebar:
    st.title("Loaded Documents")
    if st.session_state['document_list']:
        for doc in st.session_state['document_list']:
            st.write(f"- {doc}")
    else:
        st.write("No documents loaded.")

# Page title
st.title('🦜🔗 Enhanced Ask the Doc App')

# File or URL upload
uploaded_file = st.file_uploader('Upload a PDF document', type='pdf')
uploaded_url = st.text_input('Enter a website URL (optional)')

# Load documents
documents = []
if uploaded_file:
    documents = load_document(file=uploaded_file)
elif uploaded_url:
    documents = load_document(url=uploaded_url)

# Query input
query_text = st.text_input(
    'Enter your question:',
    placeholder='Ask something about the loaded documents.',
    disabled=not documents
)

# Form input and query
result = []
with st.form('query_form', clear_on_submit=True):
    openai_api_key = st.text_input(
        'OpenAI API Key',
        type='password',
        disabled=not (documents and query_text)
    )
    submitted = st.form_submit_button(
        'Submit',
        disabled=not (documents and query_text)
    )
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Generating response...'):
            response = generate_response(documents, openai_api_key, query_text)
            result.append(response)
            del openai_api_key

if len(result):
    st.info(response)
