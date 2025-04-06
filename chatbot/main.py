
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from dotenv import load_dotenv

load_dotenv()

class PDFHandler:
    @staticmethod
    def extract_pdf_content(pdf_files):
        try:
            document_text = ""
            for pdf in pdf_files:
                reader = PdfReader(pdf)
                for page in reader.pages:
                    document_text += page.extract_text()
                return document_text
        except Exception as error:
            return "Error occurred."
    
    @staticmethod
    def split_text_into_chunks(text):
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_text(text)
    
    @staticmethod
    def build_Vector_store(chunks):

        try:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vector_db = FAISS.from_texts(chunks, embedding=embeddings)
            vector_db.save_local("faiss_index")
            return True
        except Exception as error:
            return False
    

class RAGChatbot:

    def __init__(self):
        self.pdf_processor = PDFHandler()

    
    