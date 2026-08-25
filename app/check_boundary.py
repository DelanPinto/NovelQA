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

    documents.append(text)


for index in [0, 11, 12, 15, 16]:

    print("\n" + "=" * 70)
    print(f"DOCUMENT {index}")
    print("=" * 70)

    text = documents[index]

    print("\n--- START ---")
    print(text[:500])

    print("\n--- END ---")
    print(text[-500:])