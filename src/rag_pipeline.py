# rag_pipeline.py
import os
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
import warnings
warnings.filterwarnings('ignore')

os.environ["DEEPSEEK_API_KEY"] = "sk-cdfad4eedd844539824a9bd2744caa5c"

# Create retriever with metadata filtering
def get_retriever(vectorstore, question):
    """Smart retriever that filters based on question intent"""

    question_lower = question.lower()

    # Detect intent and apply filters
    if any(word in question_lower for word in ['fdh', 'foreign', 'overseas', 'imported', 'live-in', 'outside hong kong']):
        # Filter: only FDH documents
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 4,
                "filter": {"category": "FDH"}  # Only FDH docs
            }
        )
    elif any(word in question_lower for word in ['agency', 'recruitment', 'ea', 'commission']):
        # Filter: employment agency docs
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 4,
                "filter": {"category": "employment_agencies"}
            }
        )
    elif any(word in question_lower for word in ['wage', 'minimum wage', 'smw', 'hourly']):
        # Filter: wage docs
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 3,
                "filter": {"category": "wage_protection"}
            }
        )
    else:
        # General: no filter
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    return retriever

# Loading vector store
print("Loading vectorstore")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': 'cpu'}
)
vectorstore = FAISS.load_local(
    "../vectorstore/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Initializing Deepseek (API)
print("Initializing Deepseek")
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
    temperature=0.3,  # Lower to keep it factual
    max_tokens=512,
)

# Given prompt
prompt = ChatPromptTemplate.from_template("""
You are a Hong Kong labor law expert. Answer the question based ONLY on the context below.

IMPORTANT RULES:
1. If the context contains relevant information, provide the answer with specific details (numbers, rates, conditions).
2. If the context has partial information, answer with what you find and note any limitations.
3. Only say "The provided documents don't contain information about [topic]" if the context is completely irrelevant.
4. Always cite which document(s) your answer comes from.
5. Distinguish between Foreign Domestic Helpers (FDH) and other domestic workers when relevant.
6. Quote specific numbers/rates when available (e.g., "$43.1/hour", "1 rest day per 7 days").

Context: {context}

Question: {question}

Answer (be direct and specific):""")

#A test function
def ask(question):
    retriever = get_retriever(vectorstore, question)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    result = qa_chain.invoke({"query": question})
    print(f"Question: {question}")
    print(f"Answer: {result['result']}")
    print(f"Sources:")
    for i, doc in enumerate(result['source_documents'], 1):
        print(f"{i}. {doc.metadata.get('source', 'Unknown')} (Category: {doc.metadata.get('category', 'N/A')})")
    return result

# Small interactive mode
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LABOR POBOT - RAG CHATBOT (DeepSeek API)")
    print("=" * 60)

    # Test queries
    test_queries = [
        "What are the rights of domestic workers?",
        "What is the minimum wage?",
        "Do domestic workers get rest days?"
    ]

    for q in test_queries:
        ask(q)
        print("-" * 60)

    # Interactive loop
    print("\nInteractive mode (type 'quit' to exit)")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['quit', 'exit']:
            break
        if user_input:
            ask(user_input)