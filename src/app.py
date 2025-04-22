import os
import chromadb
from chromadb.utils import embedding_functions as ef
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from openai import OpenAI
from typing import List
from prompt import PROMPT


def chunk_text(path: str, chunk_size: int=512, chunk_overlap: int=50) -> List[str]:
    """
    Splits the text into chunks of specified size with a specified overlap.
    """
    chunks = []
    loader = UnstructuredMarkdownLoader(path)
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Header 5")
    ]
    text_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )
    documents = loader.load()
    for doc in documents:   
        text = doc.page_content
        split_texts = text_splitter.split_text(text)
        # Further split the text into smaller chunks
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        split_texts = recursive_splitter.split_documents(split_texts)
        # Add the split texts to the chunks list        
        chunks.extend(split_texts)
    # Remove any empty strings from the chunks
    # chunks = [chunk for chunk in chunks if chunk.strip()]
    return chunks


def create_chroma_collection() -> chromadb.Collection:
    """
    Creates a ChromaDB client with the specified settings.
    """
    chroma_client = chromadb.PersistentClient("chromadb")
    embedding_function = ef.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-large"
    )
    collection = chroma_client.get_or_create_collection(
        name="recipes_for_science",
        embedding_function=embedding_function
    )

    return collection


def add_documents_to_collection(collection: chromadb.Collection, documents: List[str]) -> chromadb.Collection:
    """
    Adds documents to the ChromaDB collection.
    """
    for ix, doc in enumerate(documents):
        collection.add(documents=[doc.page_content], ids=[str(ix)])
    print(f"Added {len(documents)} documents to the collection.")
    return collection


def query_collection(collection: chromadb.Collection, query: str, n_results: int = 5) -> List[str]:
    """
    Queries the ChromaDB collection with the specified query.
    """
    results = collection.query(
        query_texts=[query], 
        n_results=n_results
    )
    
    return results


def get_chatgpt_response(client: OpenAI, query: str, context: List[str]) -> str:
    """
    Gets a response from GPT-4o based on the query and context.
    """
    # Combine context documents into a single string
    context_text = "\n\n".join(context)
    
    # Create a prompt for GPT-4o
    prompt = PROMPT.format(
        context=context_text,
        query=query
    )
   
    
    # Call the OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    
    return response.choices[0].message.content


def chat_with_knowledge_base(client: OpenAI, collection: chromadb.Collection):
    """
    Interactive chat loop with the knowledge base.
    """
    print("Welcome to the knowledge base chat! Type 'exit' to quit.")
    
    while True:
        user_query = input("\nYour question: ")
        
        if user_query.lower() in ['exit', 'quit']:
            print("Thank you for using the knowledge base chat. Goodbye!")
            break
        
        # Query the collection
        results = query_collection(collection, user_query)
        print(results)
        
        # Extract the documents from the results
        documents = results['documents'][0] if results['documents'] else []
        
        if not documents:
            print("No relevant information found in the knowledge base.")
            continue
        
        # Get response from GPT-4o
        response = get_chatgpt_response(client, user_query, documents)
        
        # Print the response
        print("\nAnswer:", response)


if __name__ == "__main__":
    load_dotenv()
    
    import logging
    log = logging.getLogger(__name__)
    logging.basicConfig(filename='app.log', level=logging.INFO)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    
    client = OpenAI(api_key=openai_api_key)
    
    # Check if collection already exists
    try:
        collection = create_chroma_collection()
        
        # Only load data if collection is empty
        if collection.count() == 0:
            print("Loading text data and creating embeddings...")
            
            chunks = chunk_text("data/complete_text.md")
            collection = add_documents_to_collection(collection, chunks)
        else:
            print(f"Collection already has {collection.count()} documents.")
    
        # Start the chat interface
        chat_with_knowledge_base(client, collection)
    
    except Exception as e:
        log.exception("An error occurred: %s", e)