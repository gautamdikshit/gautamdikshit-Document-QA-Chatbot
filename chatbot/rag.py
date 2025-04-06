from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

import os
# os.environ["GROQ_API_KEY"]="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"


load_dotenv()

# load the document
loader = TextLoader("chatbot/speech.txt")
text_document = loader.load()

# convert into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(text_document)

# vector embedding and vector space
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Local model
db = Chroma.from_documents(documents, embedding_model)

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3}
)

llm = ChatGroq(model="llama3-8b-8192", groq_api_key="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")

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# create_history_aware_retriever
history_aware_retriever = create_history_aware_retriever(
    llm, 
    retriever, 
    contextualize_q_prompt
)

qa_system_prompt = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {input} 
Context: {context} 
Answer:
"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

# create a retrieval chain 
qa_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

def continual_chat():
    print("Start chatting with the AI! Type 'exit' to end the conversation.")
    chat_history = []
    while True:
        query = input("You: ")
        if query.lower() == "q":
            break
        result = rag_chain.invoke({"input":query, "chat_history": chat_history})
        print(f"AI: {result['answer']}")
        
        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=result["answer"]))


if __name__ == "__main__":
    continual_chat()
