NovelQA - RAG Question Answering System

A Retrieval-Augmented Generation (RAG) application that answers questions about The Great Gatsby by F. Scott Fitzgerald.

The system uses semantic search and vector embeddings to retrieve relevant passages from the novel and provides them as context to a Large Language Model (LLM) to generate grounded answers.

Features
Document ingestion and text chunking
Text embeddings using Google Gemini
Vector storage using ChromaDB
Semantic similarity search
Distance-based result filtering
Chapter diversity during retrieval
Context-grounded answer generation
Unsupported question handling
Retrieval evaluation
Answer evaluation
Architecture
Document Processing Pipeline
                 ┌──────────────────┐
                 │   Novel Text     │
                 │ The Great Gatsby │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Text Processing  │
                 │   & Chunking     │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Gemini Embeddings│
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │    ChromaDB      │
                 │ Vector Database  │
                 └──────────────────┘
Question Answering Pipeline
                 USER QUESTION
                       │
                       ▼
              ┌───────────────────┐
              │ Question Embedding │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │ Semantic Retrieval │
              └─────────┬─────────┘
                        │
                        ▼
              ┌─────────────────────────┐
              │ Distance Filtering &    │
              │ Chapter Diversity       │
              └─────────┬───────────────┘
                        │
                        ▼
              ┌───────────────────┐
              │ Retrieved Context │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │    Gemini LLM     │
              └─────────┬─────────┘
                        │
                        ▼
                   FINAL ANSWER
How It Works
1. Document Ingestion

The novel is processed and divided into smaller text chunks.

Each chunk contains metadata such as:

Chapter number
Chunk ID

Chunking allows the system to retrieve only relevant portions of the document instead of sending the entire novel to the language model.

2. Embeddings

Each text chunk is converted into a numerical vector using the Gemini embedding model:

gemini-embedding-001

Embeddings represent the semantic meaning of text.

For example, questions with different wording but similar meaning can produce similar vector representations.

3. Vector Database

The generated embeddings are stored in ChromaDB.

Each stored record contains:

Text Chunk
     +
Embedding Vector
     +
Metadata

This allows efficient semantic similarity search.

4. Retrieval

When a user asks a question:

The question is converted into an embedding.
ChromaDB searches for similar document embeddings.
Multiple candidate chunks are retrieved.
Results beyond the configured distance threshold are filtered out.
Chapter diversity is applied to avoid returning only chunks from the same chapter.
The top relevant results are returned.

The current retrieval configuration uses:

TOP_K = 3
CANDIDATE_K = 10
MAX_DISTANCE = 0.65
5. Answer Generation

The retrieved passages are combined into a context.

The context and user question are then sent to Gemini with instructions to:

Use only the provided context
Avoid using outside knowledge
Avoid inventing facts
Clearly state when the retrieved context does not contain enough information

This helps reduce hallucinations and keeps answers grounded in the retrieved source material.

Project Structure
NovelQA/
│
├── app/
│   ├── ingest.py
│   ├── retrieval.py
│   ├── rag.py
│   ├── evaluate_retrieval.py
│   └── evaluate_answers.py
│
├── data/
│   ├── raw/
│   │   └── great_gatsby.txt
│   │
│   └── evaluation/
│       ├── retrieval.json
│       └── answers.json
│
├── chroma_db/
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md

The exact structure may vary depending on the local project setup.

Technologies Used
Python
Google Gemini API
Google Gen AI SDK
ChromaDB
Vector Embeddings
Retrieval-Augmented Generation (RAG)
Installation
1. Clone the Repository
git clone <your-repository-url>
cd NovelQA
2. Create a Virtual Environment
Windows
python -m venv .venv

Activate the environment:

.venv\Scripts\activate
3. Install Dependencies
pip install -r requirements.txt
4. Configure the Gemini API Key

Create a .env file in the project root:

GEMINI_API_KEY=your_api_key_here

Make sure .env is included in .gitignore.

Example .gitignore:

.venv/
.env
__pycache__/
*.pyc
Note

Whether chroma_db/ should be included in .gitignore depends on your preference.

For a small learning project, you can exclude it and require users to run ingestion locally:

chroma_db/
Building the Vector Database

Run the ingestion script:

python app/ingest.py

This will:

Load the novel
Split it into chunks
Generate embeddings
Store the embeddings in ChromaDB
Running the RAG Application

Run:

python app/rag.py

Example:

Enter your question: Why does Gatsby throw parties?

Example output:

ANSWER

Based on the provided context, Gatsby throws parties because
he hopes Daisy will eventually wander into one of them.

The application also displays the retrieved source chapters.

Retrieval Evaluation

The project includes a retrieval evaluation script.

Run:

python app/evaluate_retrieval.py

The evaluation checks whether the retrieval system returns passages from the expected chapters.

Metrics include:

Retrieval success
Recall
Unsupported question rejection
Example Results

Results obtained using the current small evaluation dataset:

Retrieval success rate: 100%
Average Recall: 1.00
Unsupported rejection rate: 100%
Overall evaluation success: 100%

These results are based on a small, manually created evaluation dataset and should not be interpreted as a comprehensive benchmark.

Answer Evaluation

The project also includes answer evaluation.

Run:

python app/evaluate_answers.py

The evaluation process:

Retrieves relevant context
Generates an answer using Gemini
Compares the generated answer against manually defined required facts
Checks whether unsupported questions are correctly rejected

The answer evaluation makes multiple Gemini API requests and may be affected by:

API quotas
Requests-per-minute limits
Daily request limits

If the evaluation stops because of a 429 RESOURCE_EXHAUSTED error, the issue is related to API quota limits rather than the retrieval pipeline itself.

Example Questions
Supported Questions
Why does Gatsby throw parties?
Who is Jay Gatsby?
What is Gatsby's relationship with Daisy?
Why does Nick move to West Egg?
Unsupported Question
What was Gatsby's favorite meal?

For unsupported questions, the system should respond that there is insufficient information in the retrieved passages.

Key RAG Concepts Demonstrated
Embeddings

Text is converted into numerical vectors that represent semantic meaning.

Vector Search

The question embedding is compared against stored document embeddings to find semantically relevant passages.

Top-K Retrieval

The system retrieves multiple relevant chunks instead of relying on a single result.

Distance Thresholding

Chunks with a distance greater than the configured threshold are excluded.

Metadata

Each chunk stores metadata such as chapter information.

Context Grounding

The LLM is instructed to answer using only the retrieved context.

Unsupported Query Handling

If no sufficiently relevant passages are found, the system avoids generating an unsupported answer.

RAG Evaluation

Both retrieval and generated answers are evaluated using small, manually created evaluation datasets.

Limitations

This is a learning project and has several limitations:

The evaluation datasets are small.
Retrieval quality depends on the chunking strategy and embedding quality.
The distance threshold is manually configured.
The LLM may still produce imperfect answers.
API quotas and rate limits can affect evaluation runs.
The system is designed for a single novel rather than a large document collection.
Chapter diversity is a simple heuristic and may not always select the optimal context.
Future Improvements

Possible improvements include:

Support for multiple books and documents
Hybrid search using keyword and vector search
Reranking retrieved chunks
Improved chunking strategies
Web-based user interface
More comprehensive evaluation datasets
Automated RAG evaluation metrics
Conversation memory
Source citations in generated answers
What I Learned

Through this project, I learned the foundations of building a Retrieval-Augmented Generation system, including:

Document ingestion
Text chunking
Generating embeddings
Vector databases
Semantic search
Similarity filtering
Metadata-based retrieval
Context construction
LLM-based answer generation
Hallucination reduction through context grounding
Retrieval evaluation
Answer evaluation
Handling API quotas and rate limits
Author

Delan Shawn Pinto