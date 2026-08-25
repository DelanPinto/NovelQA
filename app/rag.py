import chromadb

from google import genai


# --------------------------------------------------
# Configuration
# --------------------------------------------------

CHROMA_PATH = "chroma_db"

COLLECTION_NAME = "great_gatsby"

EMBEDDING_MODEL = "gemini-embedding-001"

GENERATION_MODEL = "gemini-3.6-flash"

TOP_K = 3

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
# Retrieve chunks
# --------------------------------------------------

def retrieve_chunks(
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

    documents = results["documents"][0]

    metadatas = results["metadatas"][0]

    distances = results["distances"][0]

    ids = results["ids"][0]

    retrieved_chunks = []

    for document, metadata, distance, chunk_id in zip(
        documents,
        metadatas,
        distances,
        ids
    ):

        # Ignore results that are too far
        # from the query.
        if distance > MAX_DISTANCE:
            continue

        retrieved_chunks.append(
            {
                "id": chunk_id,
                "text": document,
                "chapter": metadata["chapter"],
                "distance": distance
            }
        )

    return retrieved_chunks


# --------------------------------------------------
# Build context
# --------------------------------------------------

def build_context(retrieved_chunks):
    """
    Combine retrieved chunks into context
    for Gemini.
    """

    context_parts = []

    for i, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        context_parts.append(
            f"""
[Source {i} - Chapter {chunk["chapter"]}]

{chunk["text"]}
"""
        )

    return "\n".join(context_parts)


# --------------------------------------------------
# Generate answer
# --------------------------------------------------

def generate_answer(
    question,
    context
):
    """
    Generate a grounded answer using Gemini.
    """

    prompt = f"""
You are a question-answering assistant
for The Great Gatsby by F. Scott Fitzgerald.

Answer the user's question using only the
provided context from the novel.

Rules:

1. Base the answer on the provided context.

2. Do not invent facts that are not supported
   by the context.

3. If the context does not contain enough
   information to answer the question, say so
   clearly.

4. Do not use your general knowledge about
   The Great Gatsby to fill missing information.

5. Give a clear and concise answer.

6. Mention relevant chapter numbers when
   appropriate.

--------------------
CONTEXT
--------------------

{context}

--------------------
QUESTION
--------------------

{question}

--------------------
ANSWER
--------------------
"""

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt
    )

    return response.text


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("=" * 60)
    print("NOVEL QA - RAG QUESTION ANSWERING")
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
        "\nCreating question embedding..."
    )

    query_embedding = (
        create_query_embedding(
            question
        )
    )

    print(
        "Retrieving relevant passages..."
    )

    retrieved_chunks = retrieve_chunks(
        collection,
        query_embedding
    )

    # --------------------------------------------------
    # No relevant context
    # --------------------------------------------------

    if not retrieved_chunks:

        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60)

        print(
            "\nThe retrieved passages do not contain "
            "enough relevant information to answer "
            "this question from the novel."
        )

        print(
            f"\nNo retrieved chunk was within the "
            f"maximum distance of {MAX_DISTANCE:.2f}."
        )

        return

    # --------------------------------------------------
    # Build context
    # --------------------------------------------------

    context = build_context(
        retrieved_chunks
    )

    # --------------------------------------------------
    # Generate answer
    # --------------------------------------------------

    print(
        "Generating answer..."
    )

    answer = generate_answer(
        question,
        context
    )

    # --------------------------------------------------
    # Display answer
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)

    print(answer)

    # --------------------------------------------------
    # Display sources
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("RETRIEVED SOURCES")
    print("=" * 60)

    for i, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        print(
            f"\nSource {i}: "
            f"Chapter {chunk['chapter']} "
            f"(distance: "
            f"{chunk['distance']:.4f})"
        )


if __name__ == "__main__":
    main()