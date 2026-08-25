import json
import sys
from pathlib import Path

from google import genai


# --------------------------------------------------
# Project paths
# --------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent

EVALUATION_FILE = (
    PROJECT_DIR
    / "data"
    / "evaluation"
    / "answers.json"
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
# Gemini configuration
# --------------------------------------------------

GENERATION_MODEL = "gemini-3.6-flash"

client = genai.Client()


# --------------------------------------------------
# Load evaluation dataset
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
# Build context
# --------------------------------------------------

def build_context(retrieved_chunks):

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

    prompt = f"""
You are answering a question about
The Great Gatsby by F. Scott Fitzgerald.

Use ONLY the provided context.

Rules:

1. Do not use outside knowledge.

2. Do not invent facts.

3. Answer the question directly.

4. If the context does not contain enough
   information, clearly state that there is
   not enough information to answer.

5. Keep the answer concise but complete.

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

    return response.text.strip()


# --------------------------------------------------
# Evaluate a supported answer
# --------------------------------------------------

def evaluate_supported_answer(
    question,
    generated_answer,
    required_facts,
    supporting_facts
):

    required_text = "\n".join(
        f"- {fact}"
        for fact in required_facts
    )

    supporting_text = "\n".join(
        f"- {fact}"
        for fact in supporting_facts
    )

    prompt = f"""
You are evaluating the answer produced by a
Retrieval-Augmented Generation system.

Determine whether the generated answer correctly
answers the question using the required facts.

Question:
{question}

Required facts:
{required_text}

Optional supporting facts:
{supporting_text}

Generated answer:
{generated_answer}

Evaluation rules:

1. Every REQUIRED fact must be correctly represented
   in the answer.

2. Supporting facts are optional.

3. Do not require exact wording.

4. Equivalent wording is acceptable.

5. Do not penalize an answer simply because it
   contains fewer supporting details.

6. The answer must not contradict the required facts.

Return ONLY:

PASS

or

FAIL
"""

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt
    )

    result = response.text.strip().upper()

    return result.startswith("PASS")


# --------------------------------------------------
# Evaluate unsupported answer
# --------------------------------------------------

def evaluate_unsupported_answer(
    question,
    generated_answer
):

    prompt = f"""
You are evaluating a RAG question-answering system.

The question is intentionally unsupported by the
available novel corpus.

Question:
{question}

Generated answer:
{generated_answer}

The correct behavior is to acknowledge that the
available context does not contain enough information
to answer the question.

PASS if the answer clearly indicates that there is
not enough information and does not invent an answer.

FAIL if the answer confidently provides an unsupported
fact.

Return ONLY:

PASS

or

FAIL
"""

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt
    )

    result = response.text.strip().upper()

    return result.startswith("PASS")


# --------------------------------------------------
# Evaluate one question
# --------------------------------------------------

def evaluate_question(
    collection,
    question_data
):

    question = question_data["question"]

    required_facts = question_data.get(
        "required_facts",
        []
    )

    supporting_facts = question_data.get(
        "supporting_facts",
        []
    )

    supported = question_data["supported"]

    print("\n" + "=" * 70)
    print("QUESTION:")
    print(question)
    print("=" * 70)

    # --------------------------------------------------
    # Create embedding
    # --------------------------------------------------

    print("\nCreating question embedding...")

    query_embedding = create_query_embedding(
        question
    )

    # --------------------------------------------------
    # Retrieve context
    # --------------------------------------------------

    print("Retrieving relevant passages...")

    retrieved_chunks = retrieve_results(
        collection,
        query_embedding
    )

    # --------------------------------------------------
    # Generate answer
    # --------------------------------------------------

    if retrieved_chunks:

        context = build_context(
            retrieved_chunks
        )

        print("Generating answer...")

        generated_answer = generate_answer(
            question,
            context
        )

    else:

        generated_answer = (
            "The retrieved passages do not contain "
            "enough relevant information to answer "
            "this question from the novel."
        )

        print(
            "\nNo sufficiently similar passages found."
        )

    # --------------------------------------------------
    # Display generated answer
    # --------------------------------------------------

    print("\nGenerated answer:")
    print("-" * 70)
    print(generated_answer)

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------

    print("\nEvaluating answer...")

    if supported:

        passed = evaluate_supported_answer(
            question,
            generated_answer,
            required_facts,
            supporting_facts
        )

    else:

        passed = evaluate_unsupported_answer(
            question,
            generated_answer
        )

    print(
        f"Answer evaluation: "
        f"{'PASS' if passed else 'FAIL'}"
    )

    # --------------------------------------------------
    # Display retrieved sources
    # --------------------------------------------------

    if retrieved_chunks:

        print("\nRetrieved sources:")

        for index, chunk in enumerate(
            retrieved_chunks,
            start=1
        ):

            print(
                f"Source {index}: "
                f"Chapter {chunk['chapter']} "
                f"(distance: "
                f"{chunk['distance']:.4f})"
            )

    return {
        "question": question,
        "supported": supported,
        "retrieved": bool(retrieved_chunks),
        "answer": generated_answer,
        "passed": passed
    }


# --------------------------------------------------
# Print summary
# --------------------------------------------------

def print_summary(results):

    total = len(results)

    passed = sum(
        result["passed"]
        for result in results
    )

    failed = total - passed

    supported_results = [
        result
        for result in results
        if result["supported"]
    ]

    unsupported_results = [
        result
        for result in results
        if not result["supported"]
    ]

    supported_passed = sum(
        result["passed"]
        for result in supported_results
    )

    unsupported_passed = sum(
        result["passed"]
        for result in unsupported_results
    )

    # --------------------------------------------------
    # Overall accuracy
    # --------------------------------------------------

    overall_accuracy = (
        passed / total
        if total
        else 0.0
    )

    # --------------------------------------------------
    # Supported-answer accuracy
    # --------------------------------------------------

    supported_accuracy = (
        supported_passed
        / len(supported_results)
        if supported_results
        else 0.0
    )

    # --------------------------------------------------
    # Unsupported rejection accuracy
    # --------------------------------------------------

    rejection_accuracy = (
        unsupported_passed
        / len(unsupported_results)
        if unsupported_results
        else 0.0
    )

    # --------------------------------------------------
    # Print summary
    # --------------------------------------------------

    print("\n\n" + "=" * 70)
    print("ANSWER EVALUATION SUMMARY")
    print("=" * 70)

    print(
        f"\nTotal questions: {total}"
    )

    print(
        f"Passed: {passed}"
    )

    print(
        f"Failed: {failed}"
    )

    print(
        f"Overall answer accuracy: "
        f"{overall_accuracy:.2%}"
    )

    print(
        f"\nSupported questions: "
        f"{len(supported_results)}"
    )

    print(
        f"Supported-answer accuracy: "
        f"{supported_accuracy:.2%}"
    )

    print(
        f"\nUnsupported questions: "
        f"{len(unsupported_results)}"
    )

    print(
        f"Unsupported-question rejection: "
        f"{rejection_accuracy:.2%}"
    )

    print("\n" + "=" * 70)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("=" * 70)
    print("NOVEL QA - ANSWER EVALUATION")
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
    # Print summary
    # --------------------------------------------------

    print_summary(results)


# --------------------------------------------------
# Entry point
# --------------------------------------------------

if __name__ == "__main__":
    main()