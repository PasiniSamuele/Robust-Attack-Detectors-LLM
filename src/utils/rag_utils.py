from pyparsing import dbl_quoted_string
from utils.custom_grobid_parser import CustomGrobidParser
from typing import List
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders.base import BaseBlobParser


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStore
from langchain.vectorstores.chroma import Chroma
from langchain.document_loaders import WebBaseLoader
import bs4

from langchain.document_loaders.generic import GenericLoader
from langchain.docstore.document import Document
from langchain_core.embeddings import Embeddings
from langchain.text_splitter import TextSplitter, RecursiveCharacterTextSplitter
from utils.path_utils import folder_exists_and_not_empty
import os

def build_scientific_papers_loader(papers_folder:str,
                                   parser:BaseBlobParser = CustomGrobidParser,
                                   segment_sentences: bool = False)->List[Document]: 
    loader = GenericLoader.from_filesystem(
        papers_folder,
        glob="*",
        suffixes=[".pdf"],
        parser=parser(segment_sentences=segment_sentences),
    )
    docs = loader.load()
    return docs

def build_web_page_loader(url:str)->List[Document]:
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
            )
        ))
    docs = loader.load()
    return docs

def build_vectorstore(docs:List[Document],
                    embeddings:Embeddings,
                    db_store : VectorStore = Chroma,
                    db_persist_path:str = "db/chroma",
                    text_splitter : TextSplitter = RecursiveCharacterTextSplitter,
                    chunk_size:int=1000,
                    chunk_overlap:int=200):
    if folder_exists_and_not_empty(db_persist_path):
        db = db_store(persist_directory=db_persist_path, embedding_function=embeddings)
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        splits = text_splitter.split_documents(docs)
        #check if db_persist_path is an existing folder and not an empty folder
        os.makedirs(db_persist_path, exist_ok=True)
        db = db_store.from_documents(splits, embeddings, persist_directory=db_persist_path)
    return db

def build_documents_retriever(docs:List[Document],
                    embeddings:Embeddings,
                    db_store : VectorStore = Chroma,
                    db_persist_path:str = "data/db/chroma",
                    text_splitter : TextSplitter = RecursiveCharacterTextSplitter,
                    chunk_size:int=1000,
                    chunk_overlap:int=200):

    vectorstore = build_vectorstore(docs, embeddings, db_store, db_persist_path, text_splitter, chunk_size, chunk_overlap) 
    retriever = vectorstore.as_retriever()
    return retriever

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

