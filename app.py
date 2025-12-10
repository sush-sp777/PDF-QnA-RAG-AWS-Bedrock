import json
import os
import sys
import boto3
import streamlit as st

from langchain_aws import ChatBedrock  #Used for text generation (LLM model).
from langchain_aws import BedrockEmbeddings  #used for generating embeddings

#data ingestion
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate

#bedrock client
bedrock=boto3.client(service_name="bedrock-runtime")

bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",client=bedrock)

def data_ingestion(uploaded_files):
    documents=[]
    for uploaded_file in uploaded_files:
        temppdf=f"./temp_{uploaded_file.name}.pdf"
        with open(temppdf,'wb') as file:
            file.write(uploaded_file.getvalue())
            file_name=uploaded_file.name
        loader=PyPDFLoader(temppdf)
        docs=loader.load()
        documents.extend(docs)
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=700,chunk_overlap=50)
    final_document=text_splitter.split_documents(documents)
    return final_document

def get_vectorstore(final_document):
    vectorstore=FAISS.from_documents(final_document,bedrock_embeddings)
    vectorstore.save_local("faiss_index")

def get_llama_llm():
    llm=ChatBedrock(
        model_id="meta.llama3-8b-instruct-v1:0",
        client=bedrock,
        model_kwargs={"max_gen_len": 512,
        "temperature": 0.5,
        "top_p": 0.9}
        )
    return llm

def get_mistral_llm():
    llm=ChatBedrock(
        model_id="mistral.mistral-large-2402-v1:0",
        client=bedrock,
        model_kwargs={"max_tokens": 512,
        "temperature": 0.5,
        "top_p": 0.9}
        )
    return llm

prompt_template="""
Use the following pieces of context to provide a concise answer to the question.
summarize with 250 words with detailed explnations. if you don't know the answer 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question:{question}   
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Answer using the provided context."),
    ("user", prompt_template)
])
def get_retriever():
    vector=FAISS.load_local(
        "faiss_index",
        bedrock_embeddings,
        allow_dangerous_deserialization=True
    )
    return vector.as_retriever(search_kwargs={"k":3})

def run_rag(question,model="Llama"):
    retriever=get_retriever()

    docs=retriever.invoke(question)
    context="\n\n".join([d.page_content for d in docs])

    prompt_value = prompt.invoke({"context": context, "question": question})
    messages = prompt_value.to_messages()

    if model=="Llama":
        llm=get_llama_llm()
    else:
        llm=get_mistral_llm()

    response=llm.invoke(messages)
    return response.content

#streamlit UI

st.title("Document Q&A RAG with Amazon Bedrock")

uploaded_files=st.file_uploader("Choose PDF file",type="pdf",accept_multiple_files=True)

if uploaded_files:
    st.write("Processing...")
    final_documents=data_ingestion(uploaded_files)
    get_vectorstore(final_documents)
    st.success("Vectorstore Created!")

question=st.text_input("Enter your question:")

if question:
    if st.button("Llama Answer"):
        with st.spinner("Generating answer from Llama..."):
            answer = run_rag(question, model="Llama")
            st.subheader("Llama Answer:")
            st.markdown(answer)

    if st.button("Mistral Answer"):
        with st.spinner("Generating answer from Mistral..."):
            answer = run_rag(question, model="mistral")
            st.subheader("Mistral Answer:")
            st.markdown(answer)