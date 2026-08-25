import json
from pathlib import Path

import chromadb


# --------------------------------------------------
# Configuration
# --------------------------------------------------

EMBEDDINGS_PATH = Path(
    "data/processed/embeddings.json"
)

CHROMA_PATH = "chroma_db"

COLLECTION_NAME = "great_gatsby"


# --------------------------------------------------
# Load embeddings
# --------------------------------------------------

def load_embeddings(file_path):
    """
    Load embedded chunks from embeddings.json.
    """

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# --------------------------------------------------
# Create ChromaDB collection
# --------------------------------------------------

def create_collection():

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    return collection


# --------------------------------------------------
# Store embeddings
# --------------------------------------------------

def store_embeddings(
    collection,
    embedded_chunks
):
    """
    Store embeddings, text, and metadata
    in ChromaDB.
    """

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for chunk in embedded_chunks:

        ids.append(
            chunk["id"]
        )

        documents.append(
            chunk["text"]
        )

        embeddings.append(
            chunk["embedding"]
        )

        metadatas.append(
            {
                "chapter": chunk["chapter"]
            }
        )

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("=" * 60)
    print("NOVEL QA - VECTOR STORE")
    print("=" * 60)

    # Load embeddings
    embedded_chunks = load_embeddings(
        EMBEDDINGS_PATH
    )

    print(
        f"\nTotal embeddings loaded: "
        f"{len(embedded_chunks)}"
    )

    # Create ChromaDB collection
    collection = create_collection()

    print(
        f"Collection: {COLLECTION_NAME}"
    )

    # Store embeddings
    store_embeddings(
        collection,
        embedded_chunks
    )

    # Verify
    total = collection.count()

    print(
        f"\nTotal vectors stored: {total}"
    )

    print(
        "\nVector store created successfully."
    )

    print(
        f"Database location: {CHROMA_PATH}"
    )


if __name__ == "__main__":
    main()