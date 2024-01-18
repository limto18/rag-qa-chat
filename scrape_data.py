from typing import List
import bs4
import os
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.indexes import SQLRecordManager, index
from langchain_community.document_loaders import WebBaseLoader

embeddings_model = OpenAIEmbeddings()

STORAGE_PATH = 'data/chroma/'

ARTICLE_URLS = ["https://www.agenceecofin.com/metaux/1601-115256-cote-d-ivoire-un-programme-de-forage-de-3-000-m-a-commence-au-projet-de-lithium-atex",
                "https://www.agenceecofin.com/sucre/1601-115255-egypte-l-esiic-arrete-la-production-de-sucre-de-canne-a-abu-qurqas",
                "https://www.agenceecofin.com/formation/1601-115254-l-association-africtivistes-lance-une-formation-en-ligne-sur-la-cybersecurite",
                "https://www.agenceecofin.com/graphite/1601-115253-tanzanie-marula-mining-obtient-sept-licences-pour-l-exploration-du-graphite",
                "https://www.agenceecofin.com/transports/1601-115251-kenya-airways-interdite-de-voler-depuis/vers-dar-es-salaam-a-partir-du-22-janvier-2024",
                ]

def process_docs(article_urls: List, persist_directory, embeddings_model, chunk_size=1000, chunk_overlap=100):
    print("Starting to scrap ..")

    loader = WebBaseLoader(
        web_paths=article_urls,
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("itemCategory", "itemTitle", "itemDateCreated", "itemIntroText")
            )
        ),
    )

    print("After scraping Loading ..")
    docs = loader.load()
    
    print("Successfully loaded to document")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n","\n"])
    splits = text_splitter.split_documents(docs)

    # Create the storage path if it doesn't exist
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    doc_search = Chroma.from_documents(documents=splits, embedding=embeddings_model, persist_directory=persist_directory)

    print(f"Chroma vector successfully saved to path: {os.path.abspath(persist_directory)}")

    #Indexing data
    namespace = "chromadb/my_documents"
    record_manager = SQLRecordManager(
        namespace, db_url="sqlite:///record_manager_cache.sql"
    )
    record_manager.create_schema()

    index_result = index(
        docs,
        record_manager,
        doc_search,
        cleanup="incremental",
        source_id_key="source",
    )

    print(f"Indexing stats: {index_result}")

    return doc_search
if __name__ == "__main__":

    process_docs(ARTICLE_URLS, STORAGE_PATH, OpenAIEmbeddings())