import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings

load_dotenv()
google_key = os.getenv('GOOGLE_API_KEY')
em_token = os.getenv('EM_TOKEN')


def ingest_knowledge():
    embeddings = HuggingFaceEndpointEmbeddings(huggingfacehub_api_token=em_token,
                                               model='sentence-transformers/all-MiniLM-L6-v2')

    loader = TextLoader('../docs/knowledge.txt', encoding='utf-8')
    documents = loader.load()
    # chunk_size:每个小块最大字符数
    # chunk_overlap:相邻小块之间重叠的字符数
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunk_documents = text_splitter.split_documents(documents)
    db = Chroma.from_documents(chunk_documents, embeddings, persist_directory='./trisolaris_db')


if __name__ == '__main__':
    ingest_knowledge()
