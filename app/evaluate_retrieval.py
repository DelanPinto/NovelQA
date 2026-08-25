import json
import sys
from pathlib import Path


# --------------------------------------------------
# Project paths
# --------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent

EVALUATION_FILE = (
    PROJECT_DIR
    / "data"
    / "evaluation"
    / "questions.json"
)


# --------------------------------------------------
# Allow imports from app directory
# --------------------------------------------------

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


from retrieval import (
    get_collection,
    create_query_embedding,
    retrieve_results
)


# --------------------------------------------------
# Load evaluation questions
# --------------------------------------------------

def load_questions():

    if not EVALUATION_FILE.exists():

        raise FileNotFoundError(
            f"Evaluation file not found:\n"
            f"{EVALUATION_FILE}"
        )

    with open(
        EVALUATION_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# --------------------------------------------------
# Evaluate one question
# --------------------------------------------------

def evaluate_question(
    collection,
    question_data
):

    question = question_data["question"]

    expected_chapters = set(
        question_data["expected_chapters"]
    )

    print("\n" + "=" * 70)
    print("QUESTION:")
    print(question)
    print("=" * 70)

    # --------------------------------------------------
    # Create embedding
    # --------------------------------------------------

    query_embedding = create_query_embedding(
        question
    )

    # --------------------------------------------------
    # Use the SAME retrieval pipeline as RAG
    # --------------------------------------------------

    results = retrieve_results(
        collection,
        query_embedding
    )

    retrieved_chapters = {
        result["chapter"]
        for result in results
    }

    # --------------------------------------------------
    # Display expected chapters
    # --------------------------------------------------

    print(
        f"\nExpected chapters: "
        f"{sorted(expected_chapters)}"
    )

    # --------------------------------------------------
    # Display retrieved results
    # --------------------------------------------------

    print(
        f"Retrieved results: {len(results)}"
    )

    if not results:

        print("No results returned.")

    else:

        for index, result in enumerate(
            results,
            start=1
        ):

            print(
                f"\nResult {index}"
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

    # --------------------------------------------------
    # Evaluate retrieval
    # --------------------------------------------------

    if expected_chapters:

        relevant_chapters = (
            expected_chapters
            & retrieved_chapters
        )

        recall = (
            len(relevant_chapters)
            / len(expected_chapters)
        )

        success = (
            len(relevant_chapters) > 0
        )

        print(
            f"\nRelevant chapters retrieved: "
            f"{sorted(relevant_chapters)}"
        )

        print(
            f"Recall: {recall:.2f}"
        )

        print(
            f"Retrieval success: "
            f"{'YES' if success else 'NO'}"
        )

        return {
            "question": question,
            "expected": expected_chapters,
            "retrieved": retrieved_chapters,
            "recall": recall,
            "success": success,
            "unsupported": False
        }

    # --------------------------------------------------
    # Unsupported question
    # --------------------------------------------------

    correctly_rejected = (
        len(results) == 0
    )

    print(
        "\nExpected: No supporting evidence"
    )

    print(
        f"Correctly rejected: "
        f"{'YES' if correctly_rejected else 'NO'}"
    )

    return {
        "question": question,
        "expected": expected_chapters,
        "retrieved": retrieved_chapters,
        "recall": None,
        "success": correctly_rejected,
        "unsupported": True
    }


# --------------------------------------------------
# Print overall evaluation
# --------------------------------------------------

def print_summary(results):

    supported_questions = [
        result
        for result in results
        if not result["unsupported"]
    ]

    unsupported_questions = [
        result
        for result in results
        if result["unsupported"]
    ]

    successful_supported = sum(
        result["success"]
        for result in supported_questions
    )

    correctly_rejected = sum(
        result["success"]
        for result in unsupported_questions
    )

    # --------------------------------------------------
    # Average recall
    # --------------------------------------------------

    if supported_questions:

        average_recall = (
            sum(
                result["recall"]
                for result in supported_questions
            )
            / len(supported_questions)
        )

    else:

        average_recall = 0.0

    # --------------------------------------------------
    # Retrieval success
    # --------------------------------------------------

    if supported_questions:

        retrieval_success_rate = (
            successful_supported
            / len(supported_questions)
        )

    else:

        retrieval_success_rate = 0.0

    # --------------------------------------------------
    # Unsupported rejection rate
    # --------------------------------------------------

    if unsupported_questions:

        rejection_rate = (
            correctly_rejected
            / len(unsupported_questions)
        )

    else:

        rejection_rate = 0.0

    # --------------------------------------------------
    # Overall
    # --------------------------------------------------

    total_successful = (
        successful_supported
        + correctly_rejected
    )

    total_questions = len(results)

    overall_success_rate = (
        total_successful
        / total_questions
        if total_questions
        else 0.0
    )

    print("\n\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print(
        f"\nTotal questions: "
        f"{total_questions}"
    )

    print(
        f"Supported questions: "
        f"{len(supported_questions)}"
    )

    print(
        f"Unsupported questions: "
        f"{len(unsupported_questions)}"
    )

    print(
        f"\nRetrieval success rate: "
        f"{retrieval_success_rate:.2%}"
    )

    print(
        f"Average Recall: "
        f"{average_recall:.2f}"
    )

    print(
        f"Unsupported rejection rate: "
        f"{rejection_rate:.2%}"
    )

    print(
        f"Overall evaluation success: "
        f"{overall_success_rate:.2%}"
    )

    print("\n" + "=" * 70)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("=" * 70)
    print("NOVEL QA - RETRIEVAL EVALUATION")
    print("=" * 70)

    # --------------------------------------------------
    # Load evaluation dataset
    # --------------------------------------------------

    questions = load_questions()

    print(
        f"\nEvaluation questions: "
        f"{len(questions)}"
    )

    # --------------------------------------------------
    # Connect to ChromaDB
    # --------------------------------------------------

    collection = get_collection()

    print(
        f"Collection: "
        f"{collection.name}"
    )

    print(
        f"Total vectors: "
        f"{collection.count()}"
    )

    # --------------------------------------------------
    # Run evaluation
    # --------------------------------------------------

    results = []

    for question_data in questions:

        result = evaluate_question(
            collection,
            question_data
        )

        results.append(result)

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print_summary(results)


# --------------------------------------------------
# Entry point
# --------------------------------------------------

if __name__ == "__main__":
    main()