import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory

# Caching expensive setup
@st.cache_resource
def setup_chain():
    # Load and prepare data
    loader = DirectoryLoader('data/', glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': "cpu"})
    vector_store = FAISS.from_documents(chunks, embeddings)

    llm = CTransformers(
        model="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        config={'max_new_tokens': 128, 'temperature': 0.01}
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type='stuff',
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
        memory=memory
    )

    return chain

# UI Setup
st.title("Ask Your doubts ")

user_input = st.text_input("Enter a question:")

if user_input:
    try:
        chain = setup_chain()
        result = chain({"question": user_input})
        st.success(result["answer"])
    except Exception as e:
        st.error(f"Something went wrong: {e}")
