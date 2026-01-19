import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()
groq_key = os.getenv('GROQ_API_KEY')
if not groq_key :
    raise ValueError("Missing required environment variables in .env file")
os.environ['GROQ_API_KEY'] = groq_key
LLM=ChatGroq(model='llama-3.1-8b-instant',temperature=0.7)
#******************Import all nessary packages******************

prompt=ChatPromptTemplate.from_template(
   """
    Answer the questions based on the provided context only.
    Please provide the most accurate respone based on the question
    <context>
    {context}
    </context>
    Question:{input}

    """
)
def create_embedded_vector():#Store all values in session
    if "vectors" not in st.session_state:
        st.session_state.embeddings=HuggingFaceEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader("./research_papers")#Data injestion
        st.session_state.docs=st.session_state.loader.load()    
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_document=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors=Chroma.from_documents(st.session_state.final_document,st.session_state.embeddings)


if st.button("Create Embedding"):
    create_embedded_vector()
    st.write("Process End")

user_prompt=st.text_input("Enter question ")

if user_prompt:
    retriver=st.session_state.vectors.as_retriever()
    document_chain=create_stuff_documents_chain(LLM,prompt)   
    retrivel_chain=create_retrieval_chain(retriver,document_chain)
    response=retrivel_chain.invoke({"input":user_prompt})   
    st.write(response['answer'])  

    with st.expander("Document similarity Search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('------------------------')

