import chromadb

from retrieval import (
    get_collection,
    create_query_embedding,
    search_collection
)


# --------------------------------------------------
# Test questions
# --------------------------------------------------

TEST_QUESTIONS = [
    "Why does Gatsby throw parties?",
    "Who is Jay Gatsby?",
    "What is Gatsby's relationship with Daisy?",
    "Why does Nick move to West Egg?",
    "What was Gatsby's favorite meal?"
]


# --------------------------------------------------
# Evaluate one question
# --------------------------------------------------

def evaluate_question(
    collection,
    question
):

    print("\n" + "=" * 70)

    print(
        f"QUESTION:\n{question}"
    )

    print("=" * 70)

    # Create embedding
    query_embedding = (
        create_query_embedding(
            question
        )
    )

    # Retrieve
    results = search_collection(
        collection,
        query_embedding
    )

    ids = results["ids"][0]

    metadatas = results["metadatas"][0]

    distances = results["distances"][0]

    # Display results
    for i in range(len(ids)):

        print(
            f"\nResult {i + 1}"
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


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("=" * 70)
    print("NOVEL QA - RETRIEVAL EVALUATION")
    print("=" * 70)

    collection = get_collection()

    print(
        f"\nCollection: great_gatsby"
    )

    print(
        f"Total vectors: "
        f"{collection.count()}"
    )

    for question in TEST_QUESTIONS:

        evaluate_question(
            collection,
            question
        )


if __name__ == "__main__":
    main()