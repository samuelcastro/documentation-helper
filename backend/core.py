from dotenv import load_dotenv
from typing import List, Dict, Any
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever

load_dotenv()

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

INDEX_NAME = "langchain-doc-index"


def run_llm(query: str, chat_history: List[Dict[str, Any]]) -> str:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # Create Pinecone vector store to perform the similarity search
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    chat = ChatOpenAI(model="gpt-3.5-turbo", verbose=True, temperature=0)

    # Create the retrieval chain. It takes the query, then searches for the most similar chunks of text in the vector store,
    # and then passes the query and the most similar chunks to the chat model.
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    # This is the augmentation process where it uses put the things together to send to the LLM.
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    # This prompt will help with chat memory, to let LLM understand the context of the user's previous interactions.
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    # This will create a retriever that will return relevant documents based on the chat history.
    # If there is no chat history, then the original question will be sent.
    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
    )

    # Get the most relevant chunks from the vector store and put them together with the query.
    qa = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )

    # Invoke the chain with the query. The input is required by the retrieval_qa_chat_prompt
    result = qa.invoke({"input": query, "chat_history": chat_history})

    new_result = {
        "query": result["input"],
        "result": result["answer"],
        "source_documents": result["context"],
    }

    return new_result


if __name__ == "__main__":
    res = run_llm(query="What is a Langchain Chain?")
    print(res["result"])
