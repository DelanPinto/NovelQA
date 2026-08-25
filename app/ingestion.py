import json
import re
from pathlib import Path

from bs4 import BeautifulSoup
from ebooklib import epub, ITEM_DOCUMENT


# ============================================================
# CONFIGURATION
# ============================================================

EPUB_PATH = "data/books/the_great_gatsby.epub"

OUTPUT_PATH = "data/processed/chunks.json"

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 250

CHAPTER_STARTS = {
    1: 0,
    2: 12,
    3: 16,
    4: 20,
    5: 24,
    6: 30,
    7: 46,
    8: 83,
    9: 100,
}

DOCUMENT_END = 120


# ============================================================
# LOAD EPUB
# ============================================================

def load_epub(file_path):
    """
    Read the EPUB and extract text from each document section.
    """

    book = epub.read_epub(file_path)

    documents = []

    for item in book.get_items_of_type(ITEM_DOCUMENT):

        soup = BeautifulSoup(
            item.get_content(),
            "html.parser"
        )

        text = soup.get_text(
            separator="\n",
            strip=True
        )

        if not text:
            continue

        documents.append({
            "text": text,
            "source": Path(file_path).name,
            "item_name": item.get_name()
        })

    return documents


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):
    """
    Clean EPUB formatting artifacts.
    """

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove publisher introduction
    text = re.sub(
        r"Download free eBooks.*?"
        r"Then wear the gold hat",
        "Then wear the gold hat",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    # Remove publisher watermark
    text = re.sub(
        r"Free\s+eBooks\s+at\s+Planet\s+eBook\.com",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Remove repeated title
    text = re.sub(
        r"The Great Gatsby",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Remove author
    text = re.sub(
        r"(?m)^\s*By F\. Scott Fitzgerald\s*$",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Remove standalone page numbers
    text = re.sub(
        r"(?m)^\s*\d+\s*$",
        "",
        text
    )

    # Remove soft hyphen character
    text = text.replace("\u00ad", "")

    # --------------------------------------------------------
    # Fix words broken across lines/pages.
    #
    # Example:
    #
    # metropoli-
    # tan
    #
    # becomes:
    #
    # metropolitan
    # --------------------------------------------------------

    text = re.sub(
        r"([A-Za-z]+)-\s*\n\s*([a-z]+)",
        r"\1\2",
        text
    )

    # --------------------------------------------------------
    # Fix known remaining split words from this EPUB.
    # --------------------------------------------------------

    broken_words = {
        "in-clined": "inclined",
        "con-fidences": "confidences",
        "quiver-ing": "quivering",
        "metropoli-tan": "metropolitan",
        "poi-gnant": "poignant",
        "swift-ly": "swiftly",
        "cir-cumstantial": "circumstantial",
        "van-ished": "vanished",
        "cat-aracts": "cataracts",
        "pulp-less": "pulpless",
        "week-ends": "weekends",
        "vague-ly": "vaguely",
        "under-ground": "underground",
        "insidi-ous": "insidious",
        "meretri-cious": "meretricious",
        "grudg-ing": "grudging",
        "transcen-dent": "transcendent",
        "adventit-ous": "adventitious",
        "author-ity": "authority",
        "dis-gusted": "disgusted",
    }

    for broken, corrected in broken_words.items():

        text = re.sub(
            rf"\b{re.escape(broken)}\b",
            corrected,
            text
        )

    # Normalize spaces
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Remove spaces around newlines
    text = re.sub(
        r" *\n *",
        "\n",
        text
    )

    # Normalize blank lines
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# REMOVE DUPLICATE EPIGRAPH
# ============================================================

def remove_duplicate_epigraph(text):
    """
    The EPUB may contain the Gatsby epigraph twice.

    Keep only the first occurrence.
    """

    epigraph_pattern = (
        r"Then wear the gold hat, if that will move her;.*?"
        r"—THOMAS PARKE D’INVILLIERS"
    )

    matches = list(
        re.finditer(
            epigraph_pattern,
            text,
            flags=re.DOTALL
        )
    )

    if len(matches) <= 1:
        return text

    first_end = matches[0].end()

    second_start = matches[1].start()

    text = (
        text[:first_end]
        + text[second_start:]
    )

    # Remove the second occurrence
    second_match = re.search(
        epigraph_pattern,
        text[first_end:],
        flags=re.DOTALL
    )

    if second_match:
        start = first_end + second_match.start()
        end = first_end + second_match.end()

        text = (
            text[:start]
            + text[end:]
        )

    return text.strip()


# ============================================================
# EXTRACT CHAPTERS
# ============================================================

def extract_chapters(documents):
    """
    Group EPUB document sections into nine chapters.
    """

    chapters = []

    chapter_numbers = list(
        CHAPTER_STARTS.keys()
    )

    for index, chapter_number in enumerate(
        chapter_numbers
    ):

        start = CHAPTER_STARTS[
            chapter_number
        ]

        if index + 1 < len(chapter_numbers):

            next_chapter = (
                chapter_numbers[index + 1]
            )

            end = CHAPTER_STARTS[
                next_chapter
            ]

        else:

            end = DOCUMENT_END

        chapter_documents = documents[
            start:end
        ]

        chapter_text = "\n\n".join(
            document["text"]
            for document in chapter_documents
        )

        chapter_text = clean_text(
            chapter_text
        )

        if chapter_number == 1:

            chapter_text = (
                remove_duplicate_epigraph(
                    chapter_text
                )
            )

        chapters.append({
            "chapter": chapter_number,
            "text": chapter_text
        })

    return chapters


# ============================================================
# GET PARAGRAPHS
# ============================================================

def get_paragraphs(text):
    """
    Convert chapter text into paragraphs.
    """

    paragraphs = []

    raw_paragraphs = re.split(
        r"\n\s*\n",
        text
    )

    for paragraph in raw_paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        paragraph = re.sub(
            r"\s+",
            " ",
            paragraph
        )

        paragraphs.append(
            paragraph
        )

    return paragraphs


# ============================================================
# CHUNK TEXT
# ============================================================

def chunk_text(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):
    """
    Split text into paragraph-aware chunks.
    """

    paragraphs = get_paragraphs(
        text
    )

    chunks = []

    current = []
    current_length = 0

    for paragraph in paragraphs:

        paragraph_length = len(
            paragraph
        )

        separator_length = (
            2 if current else 0
        )

        proposed_length = (
            current_length
            + separator_length
            + paragraph_length
        )

        # Paragraph fits
        if proposed_length <= chunk_size:

            current.append(
                paragraph
            )

            current_length = (
                proposed_length
            )

            continue

        # Save current chunk
        if current:

            chunks.append(
                "\n\n".join(current)
            )

        # Build overlap
        overlap_paragraphs = []
        overlap_length = 0

        for previous in reversed(
            current
        ):

            length = len(
                previous
            )

            separator = (
                2 if overlap_paragraphs
                else 0
            )

            if (
                overlap_length
                + separator
                + length
                > overlap
            ):
                break

            overlap_paragraphs.insert(
                0,
                previous
            )

            overlap_length += (
                separator + length
            )

        current = (
            overlap_paragraphs
            + [paragraph]
        )

        current_length = (
            sum(
                len(p)
                for p in current
            )
            + 2 * (len(current) - 1)
        )

    # Save final chunk
    if current:

        chunks.append(
            "\n\n".join(current)
        )

    return chunks


# ============================================================
# BUILD RAG CHUNKS
# ============================================================

def build_chunks(chapters):
    """
    Create chunks with metadata.
    """

    all_chunks = []

    for chapter in chapters:

        chunks = chunk_text(
            chapter["text"]
        )

        for index, chunk in enumerate(
            chunks
        ):

            all_chunks.append({
                "chunk_id": (
                    f"chapter_{chapter['chapter']}"
                    f"_chunk_{index}"
                ),

                "chapter": chapter[
                    "chapter"
                ],

                "source": Path(
                    EPUB_PATH
                ).name,

                "text": chunk
            })

    return all_chunks


# ============================================================
# SAVE CHUNKS
# ============================================================

def save_chunks(chunks, output_path):
    """
    Save chunks as JSON.
    """

    output_file = Path(
        output_path
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("NOVEL QA - EPUB INGESTION")
    print("=" * 60)

    # Load EPUB
    documents = load_epub(
        EPUB_PATH
    )

    print(
        f"\nTotal EPUB sections: "
        f"{len(documents)}"
    )

    # Extract chapters
    chapters = extract_chapters(
        documents
    )

    print(
        f"Total chapters extracted: "
        f"{len(chapters)}"
    )

    # Build chunks
    chunks = build_chunks(
        chapters
    )

    # Display statistics
    for chapter in chapters:

        chapter_chunks = [
            chunk
            for chunk in chunks
            if chunk["chapter"]
            == chapter["chapter"]
        ]

        print("\n" + "=" * 60)

        print(
            f"CHAPTER {chapter['chapter']}"
        )

        print("=" * 60)

        print(
            f"Characters: "
            f"{len(chapter['text'])}"
        )

        print(
            f"Chunks: "
            f"{len(chapter_chunks)}"
        )

        # Preview first chunk
        if chapter_chunks:

            print("\nFirst chunk:")
            print("-" * 40)

            print(
                chapter_chunks[0]["text"][
                    :1000
                ]
            )

    # Save
    save_chunks(
        chunks,
        OUTPUT_PATH
    )

    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)

    print(
        f"Chapters: {len(chapters)}"
    )

    print(
        f"Total chunks: {len(chunks)}"
    )

    print(
        f"Chunk size: {CHUNK_SIZE}"
    )

    print(
        f"Chunk overlap: {CHUNK_OVERLAP}"
    )

    print(
        f"\nSaved chunks to:"
        f"\n{OUTPUT_PATH}"
    )

    print(
        "\nIngestion complete."
    )