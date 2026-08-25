from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup


EPUB_PATH = "data/books/the_great_gatsby.epub"


book = epub.read_epub(EPUB_PATH)

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

    documents.append({
        "name": item.get_name(),
        "text": text
    })


print(f"Total document sections: {len(documents)}")

print("\nSearching for chapters...\n")

for i, document in enumerate(documents):

    text = document["text"]

    for chapter_number in range(1, 10):

        chapter_title = f"Chapter {chapter_number}"

        if chapter_title.lower() in text.lower():

            print(
                f"Found {chapter_title} "
                f"in document {i}: "
                f"{document['name']}"
            )