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

# Initialize persistent storage for document names, query history, and API key
if 'document_list' not in st.session_state:
    st.session_state['document_list'] = []
if 'query_history' not in st.session_state:
    st.session_state['query_history'] = []
if 'api_key' not in st.session_state:
    st.session_state['api_key'] = ''

def add_to_sidebar(doc_name):
    """Update the sidebar with the new document."""
    if doc_name not in st.session_state['document_list']:
        st.session_state['document_list'].append(doc_name)

def load_document(file=None, url=None):
    """Load a document from a file or URL."""
    if file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        add_to_sidebar(file.name)
        return documents
    elif url is not None:
        loader = WebBaseLoader(url)
        documents = loader.load()
        add_to_sidebar(url)
        return documents

def generate_response(documents, openai_api_key, query_text):
    """Generate a response from the loaded documents."""
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = Chroma.from_documents(texts, embeddings, persist_directory="chromadb_storage")
    db.persist()
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=openai_api_key),
        chain_type='stuff',
        retriever=retriever
    )
    return qa.run(query_text)

# Sidebar
with st.sidebar:
    # Logo
    st.image("https://lwfiles.mycourse.app/65a58160c1646a4dce257fac-public/a82c64f84b9bb42db4e72d0d673a50d0.png", use_column_width=True)

    # API Key Input
    st.session_state['api_key'] = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="Enter your OpenAI API key",
    )

    # Preloaded Documents
    st.write("**Preloaded Documents**")
    if st.button("Load Sample PDF"):
        documents = load_document(url="https://example.com/sample.pdf")  # Replace with a real URL
        st.session_state['document_list'].append("Sample PDF")
    if st.button("Load Sample Website"):
        documents = load_document(url="https://example.com")  # Replace with a real URL
        st.session_state['document_list'].append("Sample Website")

    # Query History
    st.write("**Query History**")
    for i, (query, response) in enumerate(st.session_state['query_history']):
        st.write(f"{i + 1}. {query} → {response}")

    # Export Queries
    if st.button("Export Query History"):
        history_text = "\n".join([f"{query} → {response}" for query, response in st.session_state['query_history']])
        st.download_button(
            label="Download History",
            data=history_text,
            file_name="query_history.txt",
            mime="text/plain"
        )

# Main App
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
if st.session_state['api_key'] and query_text and documents:
    with st.spinner('Generating response...'):
        response = generate_response(documents, st.session_state['api_key'], query_text)
        st.session_state['query_history'].append((query_text, response))
        st.write("**Response:**", response)
