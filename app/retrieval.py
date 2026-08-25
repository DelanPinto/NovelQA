from google import genai
import chromadb


# --------------------------------------------------
# Configuration
# --------------------------------------------------

CHROMA_PATH = "chroma_db"

COLLECTION_NAME = "great_gatsby"

EMBEDDING_MODEL = "gemini-embedding-001"

TOP_K = 3

# Initial experimental threshold.
# Lower distance = greater similarity.
MAX_DISTANCE = 0.65


# --------------------------------------------------
# Gemini client
# --------------------------------------------------

client = genai.Client()


# --------------------------------------------------
# Connect to ChromaDB
# --------------------------------------------------

def get_collection():
    """
    Connect to the existing ChromaDB collection.
    """

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
    Retrieve the closest chunks from ChromaDB.
    """

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results


# --------------------------------------------------
# Filter results
# --------------------------------------------------

def filter_results(
    results,
    max_distance=MAX_DISTANCE
):
    """
    Remove results whose distance is greater
    than the configured threshold.
    """

    documents = results["documents"][0]

    metadatas = results["metadatas"][0]

    distances = results["distances"][0]

    ids = results["ids"][0]

    filtered = []

    for document, metadata, distance, chunk_id in zip(
        documents,
        metadatas,
        distances,
        ids
    ):

        if distance <= max_distance:

            filtered.append(
                {
                    "id": chunk_id,
                    "text": document,
                    "chapter": metadata["chapter"],
                    "distance": distance
                }
            )

    return filtered


# --------------------------------------------------
# Display results
# --------------------------------------------------

def display_results(
    question,
    results,
    filtered_results
):
    """
    Display original and filtered retrieval results.
    """

    print("\n" + "=" * 60)
    print("RETRIEVAL RESULTS")
    print("=" * 60)

    print(
        f"\nQuestion:\n{question}"
    )

    print(
        f"\nConfigured maximum distance: "
        f"{MAX_DISTANCE:.2f}"
    )

    print(
        f"Original results: "
        f"{len(results['ids'][0])}"
    )

    print(
        f"Results after filtering: "
        f"{len(filtered_results)}"
    )

    print("\n" + "-" * 60)
    print("FILTERED RESULTS")
    print("-" * 60)

    for i, result in enumerate(
        filtered_results,
        start=1
    ):

        print(
            f"\nResult {i}"
        )

        print(
            f"ID: {result['id']}"
        )

        print(
            f"Chapter: {result['chapter']}"
        )

        print(
            f"Distance: "
            f"{result['distance']:.4f}"
        )

        print("\nText:")

        print(
            result["text"]
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

    collection = get_collection()

    print(
        f"\nSearching collection: "
        f"{COLLECTION_NAME}"
    )

    print(
        "Creating question embedding..."
    )

    query_embedding = (
        create_query_embedding(
            question
        )
    )

    print(
        "Searching for relevant chunks..."
    )

    results = search_collection(
        collection,
        query_embedding
    )

    filtered_results = filter_results(
        results
    )

    display_results(
        question,
        results,
        filtered_results
    )


if __name__ == "__main__":
    main()