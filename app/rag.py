import sys
from pathlib import Path

from google import genai


# --------------------------------------------------
# Allow imports from the app directory
# --------------------------------------------------

APP_DIR = Path(__file__).resolve().parent

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


from retrieval import (
    get_collection,
    create_query_embedding,
    retrieve_results
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

GENERATION_MODEL = "gemini-3.6-flash"


# --------------------------------------------------
# Gemini client
# --------------------------------------------------

client = genai.Client()


# --------------------------------------------------
# Build context
# --------------------------------------------------

def build_context(retrieved_chunks):
    """
    Combine retrieved chunks into a context string
    that can be provided to the generation model.
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
    Generate an answer using only the retrieved
    context from the novel.
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
# Display sources
# --------------------------------------------------

def display_sources(retrieved_chunks):
    """
    Display the sources used to generate the answer.
    """

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

    # --------------------------------------------------
    # Connect to vector database
    # --------------------------------------------------

    collection = get_collection()

    # --------------------------------------------------
    # Create question embedding
    # --------------------------------------------------

    print(
        "\nCreating question embedding..."
    )

    query_embedding = (
        create_query_embedding(
            question
        )
    )

    # --------------------------------------------------
    # Retrieve relevant chunks
    # --------------------------------------------------

    print(
        "Retrieving relevant passages..."
    )

    retrieved_chunks = retrieve_results(
        collection,
        query_embedding
    )

    # --------------------------------------------------
    # Handle no relevant results
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
            "\nNo sufficiently similar passages "
            "were found."
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

    display_sources(
        retrieved_chunks
    )


# --------------------------------------------------
# Entry point
# --------------------------------------------------

if __name__ == "__main__":
    main()