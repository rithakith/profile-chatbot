import google.generativeai as genai
import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from fastapi import FastAPI, Depends
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from fastapi.middleware.cors import CORSMiddleware
from langchain.llms.base import LLM

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class GeminiLLM(LLM):
    def __init__(self, model):
        self.model = model

    def _call(self, prompt):
        return self.model.generate_content(prompt).text

    def _llm_type(self):
        return "Gemini"

    def generate(self, prompt):
        return self._call(prompt)

def get_embeddings():
    return HuggingFaceBgeEmbeddings(
        model_name='BAAI/bge-small-en-v1.5',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )

def load_vector_store(embeddings):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    file_path = os.path.join( 'me.txt')
    
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    chunks = text_splitter.split_text(text)
    return FAISS.from_texts(texts=chunks, embedding=embeddings)

@app.get("/chat/")
async def chat(query: str, embeddings=Depends(get_embeddings)):
    vector_store = load_vector_store(embeddings)
    
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.1}
    )
    search_results = retriever.invoke(query)
    retrieved_chunks = [result.page_content for result in search_results]

    response_text = (
        f"be friendly. This question is asked by a user who has visited my portfolio. "
        f"Based on the following content: {retrieved_chunks}, answer the question: {query}"
        if retrieved_chunks else
        f"be friendly. This question is asked by a user who has visited my portfolio. "
        f"If no relevant content found, here is an answer based on general knowledge: {query}"
    )

    response = genai.GenerativeModel('gemini-1.5-flash').generate_content(response_text)
    return response.text
