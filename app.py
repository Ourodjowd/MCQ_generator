import streamlit as st
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from langchain.memory import ConversationBufferMemory   
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.document_loaders import TextLoader, PDFMinerLoader
import numpy as np
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import tempfile
from PyPDF2 import PdfReader
import io
from langchain.schema import Document 

# Configuration de la page Streamlit
st.set_page_config(page_title="Systèm RAG with MCQs", page_icon="📚")
st.title("RAG-based Question-Answer System ")

# Désactiver le parallélisme des tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Charger les variables d'environnement
load_dotenv()
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_base = os.getenv("AZURE_OPENAI_API_BASE")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# Initialiser le modèle d'embedding
embedding_model =AzureOpenAIEmbeddings(
    azure_endpoint = os.getenv("EMBEDDING_API_BASE"),
    openai_api_key = os.getenv("EMBEDDING_API_KEY"),
    deployment = os.getenv("EMBEDDING_DEPLOYMENT_NAME"),
    chunk_size=10,
)

# Initialiser le LLM
llm = AzureChatOpenAI(

    azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    model_name="gpt-4o",
    temperature=0.9,
    max_tokens=300,
)

def load_pdf_file(pdf_file):
    """Load text from a PDF file."""
    try:
        # Lire le PDF depuis le fichier uploadé
        pdf_reader = PdfReader(io.BytesIO(pdf_file.getvalue()))
        text = ""
        
        # Extraire le texte de chaque page
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
            
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

# Section pour télécharger le fichier
st.header("1. Download the document")
uploaded_file = st.file_uploader("Choose a pdf file", type=['pdf'])

if uploaded_file is not None:
    text = load_pdf_file(uploaded_file)
        
    if text:
        st.success("PDF loaded successfully!")

        # Afficher un aperçu du texte
        with st.expander("Preview of PDF contents"):
            st.text(text[:500] + "...")

        try:
            # Convertir le texte brut en liste de Document
            documents = [Document(page_content=text)]

            # Splitter le texte
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
            split_docs = text_splitter.split_documents(documents)
                        

            st.success(f"Document loaded successfully! Number of segments: {len(split_docs)}")

            # Créer la base de données vectorielle
            vector_db = FAISS.from_documents(split_docs, embedding_model)
            retriever = vector_db.as_retriever()
            
            # Créer la chaîne QA
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
            
            # Section pour poser des questions
            #st.header("2. Poser une question")
            #query = st.text_area(
                #"Entrez votre question ici:",
                #height=100,
                #placeholder="Exemple: Qui a gagné le match final? ou 'give 5 MCQs with options and answers'"
            #)'''
            
            
            with st.spinner("Generating ..."):
                
                # Générer des MCQs
                prompt = f"""Based on the following text, generate 5 short answer questions with their correct answers.
                
                """
                response = qa_chain.invoke(prompt)
                st.subheader("Generated Short Answer Questions")
                st.write(response['result'])
            

            
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
