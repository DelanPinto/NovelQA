from pathlib import Path

import chromadb
from google import genai


# --------------------------------------------------
# Configuration
# --------------------------------------------------

CHROMA_PATH = "chroma_db"

COLLECTION_NAME = "great_gatsby"

EMBEDDING_MODEL = "gemini-embedding-001"

TOP_K = 3


# --------------------------------------------------
# Gemini client
# --------------------------------------------------

client = genai.Client()


# --------------------------------------------------
# Connect to ChromaDB
# --------------------------------------------------

def get_collection():

    chroma_client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = chroma_client.get_collection(
        name=COLLECTION_NAME
    )

    return collection


# --------------------------------------------------
# Create query embedding
# --------------------------------------------------

def create_query_embedding(question):
    """
    Convert the user's question into an embedding.
    """

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=question
    )

    return response.embeddings[0].values


# --------------------------------------------------
# Search ChromaDB
# --------------------------------------------------

def search_collection(
    collection,
    query_embedding,
    top_k=TOP_K
):
    """
    Find the chunks most similar to the query.
    """

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results


# --------------------------------------------------
# Display results
# --------------------------------------------------

def display_results(
    question,
    results
):

    print("\n" + "=" * 60)
    print("RETRIEVAL RESULTS")
    print("=" * 60)

    print(
        f"\nQuestion:\n{question}"
    )

    documents = results["documents"][0]

    metadatas = results["metadatas"][0]

    distances = results["distances"][0]

    ids = results["ids"][0]

    for i in range(len(documents)):

        print("\n" + "-" * 60)

        print(
            f"Result {i + 1}"
        )

        print(
            f"ID: {ids[i]}"
        )

        print(
            f"Chapter: "
            f"{metadatas[i]['chapter']}"
        )

        print(
            f"Distance: "
            f"{distances[i]:.4f}"
        )

        print("\nText:")

        print(
            documents[i]
        )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("=" * 60)
    print("NOVEL QA - SEMANTIC RETRIEVAL")
    print("=" * 60)

    question = input(
        "\nEnter your question: "
    ).strip()

    if not question:

        print(
            "Question cannot be empty."
        )

        return

    # Connect to ChromaDB
    collection = get_collection()

    print(
        f"\nSearching collection: "
        f"{COLLECTION_NAME}"
    )

    # Create embedding for question
    print(
        "Creating question embedding..."
    )

    query_embedding = (
        create_query_embedding(
            question
        )
    )

    # Search
    print(
        "Searching for relevant chunks..."
    )

    results = search_collection(
        collection,
        query_embedding
    )

    # Display
    display_results(
        question,
        results
    )


if __name__ == "__main__":
    main()