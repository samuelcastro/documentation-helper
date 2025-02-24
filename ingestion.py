from dotenv import load_dotenv
import os

load_dotenv()

# Split documents into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Helps building documents for Github repositories
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def ingest_docs():
    loader = ReadTheDocsLoader("langchain-docs/api.python.langchain.com/en/latest")

    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} documents")

    # Chunking
    # 1 - The rule of thumb is to send between 4 or 5 document context.
    # So let's say we're going to reseverv around 2k tokens for the context.
    # So if we have 4 documents, that means 2k / 4 = 500 tokens per context.
    # This vary, for example, if we need a short and consize answer then, we'll have
    # more tokens per context.
    # 2 - We should not limit the size of the chunks, because too small chunks will have
    # no semantic meaning, and we need semantic meaning to perform the similarity search
    # in our vector database.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Splitted into {len(documents)} chunks")

    # Embeddings
    PineconeVectorStore.from_documents(
        documents, embeddings, index_name="langchain-doc-index"
    )

    print("Documents embedded and stored in Pinecone")

if __name__ == "__main__":
    ingest_docs()