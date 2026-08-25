import json
import os
from pathlib import Path

from google import genai


# ============================================================
# CONFIGURATION
# ============================================================

CHUNKS_PATH = "data/processed/chunks.json"
OUTPUT_PATH = "data/processed/embeddings.json"

EMBEDDING_MODEL = "gemini-embedding-001"


# ============================================================
# LOAD API KEY
# ============================================================

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set."
    )


# ============================================================
# INITIALIZE GOOGLE GENAI CLIENT
# ============================================================

client = genai.Client(
    api_key=API_KEY
)


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks(file_path):
    """
    Load processed chunks from chunks.json.
    """

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# CREATE EMBEDDING
# ============================================================

def create_embedding(text):
    """
    Create an embedding vector for a piece of text.
    """

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text
    )

    return response.embeddings[0].values


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("NOVEL QA - EMBEDDING GENERATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load chunks
    # --------------------------------------------------------

    chunks = load_chunks(CHUNKS_PATH)

    print(f"\nTotal chunks loaded: {len(chunks)}")

    # --------------------------------------------------------
    # Create embeddings
    # --------------------------------------------------------

    embedded_chunks = []

    for index, chunk in enumerate(chunks):

        print(
            f"\nEmbedding chunk "
            f"{index + 1}/{len(chunks)}..."
        )

        embedding = create_embedding(
            chunk["text"]
        )

        embedded_chunks.append(
            {
                "id": chunk["chunk_id"],
                "chapter": chunk["chapter"],
                "text": chunk["text"],
                "embedding": embedding
            }
        )

        print(
            f"Embedding dimensions: "
            f"{len(embedding)}"
        )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    Path(
        OUTPUT_PATH
    ).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save embeddings
    # --------------------------------------------------------

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            embedded_chunks,
            file,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("EMBEDDING GENERATION COMPLETE")
    print("=" * 60)

    print(
        f"Chunks embedded: "
        f"{len(embedded_chunks)}"
    )

    print(
        f"Saved to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()