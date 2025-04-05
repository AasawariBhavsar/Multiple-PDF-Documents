from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model=genai.GenerativeModel("gemini-2.0-flash")

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks=text_splitter.split_text(text)
    return chunks

def get_vector_store(chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store=FAISS.from_texts(chunks, embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain():
    prompt_template="""
    Answer the question based on the context provided. If the answer is not in the context, say 'I don't know'.
    Context: \n{context}\n
    Question: \n{question}\n

    Answer:

    """
    model=ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=0.3)
    prompt=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain=load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db=FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs=new_db.similarity_search(user_question)
    chain=get_conversational_chain()
    response=chain.run(input_documents=docs, question=user_question,return_only_outputs=True)
    print(response)
    st.write("Reply: ",response)    

def main():
    st.set_page_config(page_title="PDF Chatbot", page_icon=":robot_face:")
    st.header("PDF Chatbot")
    user_question=st.text_input("Ask a question about the PDF document:")
    if user_question:
        user_input(user_question)
    with st.sidebar:
        st.subheader("Upload PDF Document")
        pdf_docs=st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
        if st.button("Process"):
            if pdf_docs:
                text=get_pdf_text(pdf_docs)
                chunks=get_text_chunks(text)
                vector_store=get_vector_store(chunks)
                st.success("PDF document processed successfully!")
            else:
                st.warning("Please upload a PDF document.")

if __name__=="__main__":
    main()