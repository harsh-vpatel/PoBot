# embedding_setup.py - RUN FILE FROM CLI, NOTEBOOK ONLY FOR DISPLAY (libraries in venv)
import os
import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document


# Loading chunks
def load_chunks(json_path="../data/processed/processed_chunks.json"):
    """Loads chunks saved from preprocessing step"""

    with open(json_path, 'r', encoding='utf-8') as f:
        chunk_data = json.load(f)

    # Converting back to Document objects
    documents = []
    for item in chunk_data:
        doc = Document(
            page_content=item["content"],
            metadata={
                "source": item["source"],
                "type": item.get("type", "unknown"),
                "category": item.get("category", "unknown"),
                "worker_type": item.get("worker_type", "unknown"),
                "keywords": item.get("keywords", [])
            }
        )
        documents.append(doc)

    print(f"Loaded {len(documents)} chunks from {json_path}")
    return documents


# Embeddings and vectorstore
print("Creating embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Load chunks
chunks = load_chunks()

# Build FAISS index
print("Building FAISS index")
vectorstore = FAISS.from_documents(chunks, embeddings)

# Saving
os.makedirs("../vectorstore", exist_ok=True)
vectorstore.save_local("../vectorstore/faiss_index")
print(f"Saved vectorstore with {len(chunks)} chunks")

# Creating retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Testing retrieval
def test_retrieval(query):
    results = retriever.invoke(query)
    print(f"Query: {query}")
    print(f"Retrieved {len(results)} chunks:\n")
    for i, doc in enumerate(results, 1):
        print(f"{i}. Source: {doc.metadata['source']}")
        print(f"Content: {doc.page_content[:150]}...\n")
    return results


if __name__ == "__main__":
    # Some test queries
    test_retrieval("What are the rights of domestic workers?")
    test_retrieval("What is the minimum wage?")
    test_retrieval("What are the rules for employment agencies?")