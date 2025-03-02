import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings, HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.prompts import PromptTemplate

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from transformers import AutoTokenizer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_num_tokens(text):
    embedding_model_id = 'BAAI/bge-base-en-v1.5'
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_id)
    return len(tokenizer.encode(text))

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=30,
        separators=["\n\n", "\n"],
        length_function=get_num_tokens
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks, persist_directory="vectorstore"):
    embedding_model = HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5')
    
    if os.path.exists(persist_directory):
        # Load existing vector store
        db = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
        return db
    else:
        # Create new vector store and persist it
        db = Chroma.from_texts(text_chunks, embedding_model, persist_directory=persist_directory)
        db.persist()  # Persist the vector store
        return db

def get_answer(vectorstore, question):
    model_id = "models/gemini-1.5-flash"
    llm = ChatGoogleGenerativeAI(model=model_id)

    prompt_template = """
    You are an assistant that helps the people of Turkey to learn about the constitution.
    Use the information below to answer the user's question.

    Relevant Information:
    {context}

    Question:
    {question}
    """

    prompt_generator = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 6}
    )
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()} # Steps 1, 2, 3
        | prompt_generator # Step 4
        | llm # Step 5
        | StrOutputParser() # Step 6
    )
    output = rag_chain.invoke(question)
    print(output)
    return output

def main():
    # Streamlit Interface
    st.title("Turkey Constitution Knowledge Assistant :book:")

    # Upload PDF
    uploaded_files = st.file_uploader("Upload your PDF files", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        # Extract text from PDFs
        st.write("Processing uploaded files...")
        extracted_text = get_pdf_text(uploaded_files)
        text_chunks = get_text_chunks(extracted_text)
        
        # Create or load the vectorstore
        st.write("Building or loading vector store...")
        vectorstore = get_vectorstore(text_chunks)

    # Input box for asking questions
    question = st.text_input("Ask a question based on the PDF contents:")

    if st.button("Get Answer"):
        if question:
            print("sdfgsasdfad")
            st.write("Searching for the answer...")
            answer = get_answer(vectorstore, question)
            st.write(answer)
            print(answer)
        else:
            st.write("Please enter a question.")

if __name__ == "__main__":
    main()