
import os
from langchain_groq import ChatGroq
from sympy import re
from Retrieval_pipeline import rag_retreiver
import streamlit as st 
from dotenv import load_dotenv


load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    groq_api_key = st.secrets.get("GROQ_API_KEY")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=groq_api_key,
    temperature=0.1
)




##simple rag function : retrieve context + generate response

def rag_simple(query,retriever, llm_model ,top_k = 3):
    #retrieve the context

    results = retriever.retrieve(query,top_k=top_k)

    context = "\n\n".join([doc['content'] for doc in results]) if results else " "

    if not context:
        return "No relevant context found to answer the question."

    #Generate the anser using groq llm

    prompt = f""" use the following context to anser  the question concisely. 
    context : {context}
    Question : {query} Answer"""
    

    response = llm_model.invoke(prompt)
        
    return response.content

st.title("Campus AI Helpdesk RAG System")
question = st.text_input("Enter your query here:")

if st.button("Ask") or question:
    if question.strip():
        with st.spinner("searching and generating response..."):
            answer = rag_simple(question, rag_retreiver , llm)
            st.write(answer)

    else:
        st.warning("please enter a question")
