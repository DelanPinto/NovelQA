from google import genai
import chromadb


# --------------------------------------------------
# Configuration
# --------------------------------------------------

CHROMA_PATH = "chroma_db"

COLLECTION_NAME = "great_gatsby"

EMBEDDING_MODEL = "gemini-embedding-001"

# Number of final results returned to the RAG system.
TOP_K = 3

# Retrieve more candidates initially so that
# we have enough results to choose from.
CANDIDATE_K = 10

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
    candidate_k=CANDIDATE_K
):
    """
    Retrieve a larger pool of candidate chunks.

    We retrieve more than TOP_K because some candidates
    may later be removed by distance filtering or
    chapter-diversity selection.
    """

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_k
    )

    return results


# --------------------------------------------------
# Filter by similarity distance
# --------------------------------------------------

def filter_results(
    results,
    max_distance=MAX_DISTANCE
):
    """
    Remove chunks whose distance is greater than
    the configured maximum distance.
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
# Select diverse results
# --------------------------------------------------

def select_diverse_results(
    filtered_results,
    top_k=TOP_K
):
    """
    Select results while preferring different chapters.

    The results are already ordered by distance.

    First pass:
        Select the best result from each chapter.

    Second pass:
        If we still need more results, fill the
        remaining slots with the next best chunks.
    """

    selected = []

    selected_ids = set()

    selected_chapters = set()

    # --------------------------------------------------
    # First pass: prefer different chapters
    # --------------------------------------------------

    for result in filtered_results:

        if len(selected) >= top_k:
            break

        chapter = result["chapter"]

        if chapter in selected_chapters:
            continue

        selected.append(result)

        selected_ids.add(result["id"])

        selected_chapters.add(chapter)

    # --------------------------------------------------
    # Second pass: fill remaining slots
    # --------------------------------------------------

    if len(selected) < top_k:

        for result in filtered_results:

            if len(selected) >= top_k:
                break

            if result["id"] in selected_ids:
                continue

            selected.append(result)

            selected_ids.add(result["id"])

    return selected


# --------------------------------------------------
# Retrieve final results
# --------------------------------------------------

def retrieve_results(
    collection,
    query_embedding,
    top_k=TOP_K
):
    """
    Complete retrieval pipeline:

        1. Retrieve candidates
        2. Apply distance filtering
        3. Apply chapter diversity
        4. Return final results
    """

    results = search_collection(
        collection,
        query_embedding
    )

    filtered_results = filter_results(
        results
    )

    final_results = select_diverse_results(
        filtered_results,
        top_k=top_k
    )

    return final_results


# --------------------------------------------------
# Display results
# --------------------------------------------------

def display_results(
    question,
    results
):
    """
    Display final retrieval results.
    """

    print("\n" + "=" * 60)
    print("RETRIEVAL RESULTS")
    print("=" * 60)

    print(
        f"\nQuestion:\n{question}"
    )

    print(
        f"\nFinal results: {len(results)}"
    )

    for i, result in enumerate(
        results,
        start=1
    ):

        print("\n" + "-" * 60)

        print(
            f"Result {i}"
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
        "Retrieving candidate chunks..."
    )

    results = retrieve_results(
        collection,
        query_embedding
    )

    display_results(
        question,
        results
    )


if __name__ == "__main__":
    main()