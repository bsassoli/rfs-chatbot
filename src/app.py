import os
import chromadb
import yaml
from chromadb.utils import embedding_functions as ef
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from openai import OpenAI
from typing import List, Dict, Any
from prompt import PROMPT


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads the configuration file.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def chunk_text(config: Dict[str, any]) -> List[str]:
    """
    Splits the text into chunks of specified size with a specified overlap.
    """
    chunks = []
    path_to_doc = config["document_processing"]["path_to_doc"]
    loader = UnstructuredMarkdownLoader(path_to_doc)
    chunk_size = config["document_processing"]["chunk_size"]
    chunk_overlap = config["document_processing"]["chunk_overlap"]
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Header 5"),
    ]
    text_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    documents = loader.load()
    for doc in documents:
        text = doc.page_content
        split_texts = text_splitter.split_text(text)
        # Further split the text into smaller chunks
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
        )
        split_texts = recursive_splitter.split_documents(split_texts)
        # Add the split texts to the chunks list
        chunks.extend(split_texts)

    # Remove empty strings from the chunks
    chunks = [chunk for chunk in chunks if chunk.strip()]

    return chunks


def create_chroma_collection(config: Dict[str, Any]) -> chromadb.Collection:
    """
    Creates a ChromaDB client with the specified settings.
    """
    chroma_client = chromadb.PersistentClient(config["chromadb"]["persist_directory"])
    embedding_function = ef.OpenAIEmbeddingFunction(
        api_key=config["openai"]["api_key"],
        model_name=config["openai"]["embedding_model"],
    )
    collection = chroma_client.get_or_create_collection(
        name=config["chromadb"]["collection_name"],
        embedding_function=embedding_function,
    )

    return collection


def add_documents_to_collection(
    collection: chromadb.Collection, documents: List[str]
) -> chromadb.Collection:
    """
    Adds documents to the ChromaDB collection.
    """
    for ix, doc in enumerate(documents):
        collection.add(documents=[doc.page_content], ids=[str(ix)])
    print(f"Added {len(documents)} documents to the collection.")
    return collection


def query_collection(
    config: Dict[str, Any], collection: chromadb.Collection, query: str
) -> List[str]:
    """
    Queries the ChromaDB collection with the specified query.
    """
    results = collection.query(
        query_texts=[query],
        n_results=config["query"]["n_results"],
    )

    return results


def get_chatgpt_response(
    config: Dict[str, Any], client: OpenAI, query: str, context: List[str]
) -> str:
    """
    Gets a response from GPT-4o based on the query and context.
    """
    # Combine context documents into a single string
    context_text = "\n\n".join(context)

    # Create a prompt for GPT-4o
    prompt = PROMPT.format(context=context_text, query=query)

    # Call the OpenAI API
    response = client.chat.completions.create(
        model=config["openai"]["completion_model"],
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on the provided context.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content


def chat_with_knowledge_base(
    config: Dict[str, Any], client: OpenAI, collection: chromadb.Collection
):
    """
    Interactive chat loop with the knowledge base.
    """
    print("Welcome to the knowledge base chat! Type 'exit' to quit.")

    while True:
        user_query = input("\nYour question: ")

        if user_query.lower() in ["exit", "quit"]:
            print("Thank you for using the knowledge base chat. Goodbye!")
            break

        # Query the collection
        results = query_collection(config, collection, user_query)
        print(results)

        # Extract the documents from the results
        documents = results["documents"][0] if results["documents"] else []

        if not documents:
            print("No relevant information found in the knowledge base.")
            continue

        # Get response from completion model
        print("Retrieving relevant information...")
        response = get_chatgpt_response(client, user_query, documents)

        # Print the response
        print("\nAnswer:", response)


if __name__ == "__main__":
    load_dotenv(".env", override=True)

    import logging

    log = logging.getLogger(__name__)
    logging.basicConfig(filename="app.log", level=logging.INFO)

    # Load configuration
    config_path = "config/config.yaml"
    config = load_config(config_path)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError(
            "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
        )
    else:
        print(openai_api_key)

    client = OpenAI(api_key=openai_api_key)

    # Check if collection already exists
    try:
        collection = create_chroma_collection(config=config)

        # Only load data if collection is empty
        if collection.count() == 0:
            print("Loading text data and creating embeddings...")
            chunks = chunk_text(config)
            collection = add_documents_to_collection(collection, chunks)
            print(
                f"Collection '{config['chromadb']['collection_name']}' created with {collection.count()} documents."
            )
        else:
            print(f"Collection already has {collection.count()} documents.")

        # Start the chat interface
        chat_with_knowledge_base(config, client, collection)

    except Exception as e:
        log.exception("An error occurred: %s", e)
