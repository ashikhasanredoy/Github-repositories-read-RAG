**# 🧠 GitHub Repositories Read RAG — Codebase Intelligence Assistant**



An advanced, production-grade **\*\*Code Retrieval-Augmented Generation (RAG)\*\*** system designed to ingest, parse, index, and query any public GitHub repository or local codebase with line-by-line understanding, exact file/symbol citations, real-time ChatGPT-style code streaming, and local LLM execution.



Powered by **\*\*AST-aware code chunking\*\***, **\*\*ChromaDB + BM25Plus Hybrid Retrieval (RRF)\*\***, and a **\*\*Corrective LangGraph Workflow\*\*** using local Ollama models (e.g., \`llama3.2:latest\`, \`olmo:latest\`).



\---



**## 🏗️ System Architecture & Workflow**



\`\`\`text

&#x20;                              GitHub Repository (URL or Local Directory)

&#x20;                                                 │

&#x20;                                                 ↓

&#x20;                                 ┌───────────────────────────────┐

&#x20;                                 │   [1] REPOSITORY INGESTION    │

&#x20;                                 │   • Git Shallow Cloner / Local │

&#x20;                                 │   • Extension & Ignore Filter │

&#x20;                                 └───────────────┬───────────────┘

&#x20;                                                 │

&#x20;                                                 ↓

&#x20;                                 ┌───────────────────────────────┐

&#x20;                                 │   [2] AST-AWARE CODE PARSING  │

&#x20;                                 │   • Python AST Tree Traversal │

&#x20;                                 │   • Class/Function Extractors │

&#x20;                                 │   • Exact Line Bounds & Dims  │

&#x20;                                 └───────────────┬───────────────┘

&#x20;                                                 │

&#x20;                                                 ↓

&#x20;                                 ┌───────────────────────────────┐

&#x20;                                 │      [3] DUAL HYBRID INDEX    │

&#x20;                                 │   Dense Vector + BM25 Lexical │

&#x20;                                 └───────┬───────────────┬───────┘

&#x20;                                         │               │

&#x20;                                         ↓               ↓

&#x20;                                  ChromaDB Store     BM25+ Index

&#x20;                                 (nomic-embed)    (Code Tokenizer)

&#x20;                                         │               │

&#x20;                                         └───────┬───────┘

&#x20;                                                 │

&#x20;┌────────────────────────────────────────────────┼────────────────────────────────────────┐

&#x20;│                                                ↓                                        │

&#x20;│   User Question / Snippet ──────────► [4] LANGGRAPH WORKFLOW                            │

&#x20;│                                       • Query Intent & Entity Extraction                │

&#x20;│                                       • Reciprocal Rank Fusion (RRF Search)             │

&#x20;│                                       • Relevance Grader & Self-Correction              │

&#x20;│                                       • Grounded Ollama LLM Reasoning                   │

&#x20;│                                                │                                        │

&#x20;└────────────────────────────────────────────────┼────────────────────────────────────────┘

&#x20;                                                 │

&#x20;                                                 ↓

&#x20;                                 ┌───────────────────────────────┐

&#x20;                                 │  [5] SSE REAL-TIME STREAMING  │

&#x20;                                 │  • ChatGPT-Style Code Typing  │

&#x20;                                 │  • Copy & Download (.py, etc.)│

&#x20;                                 │  • Line-by-Line AI Explainer  │

&#x20;                                 │  • Exact File & Line Citations│

&#x20;                                 └───────────────────────────────┘

\`\`\`



\---



**## 🔍 How This Project Actually Works (Under the Hood)**



Traditional document RAG systems split raw text arbitrarily every *\*N\** characters or tokens. When applied to source code, this breaks functions in half, loses class contexts, drops indentation, and misplaces variable definitions.



**\*\*GitHub Repositories Read RAG\*\*** solves this through a 6-stage engineering pipeline:



\---



**### 1. Repository Ingestion & Smart Filtering (\`src/code_rag/core/cloner.py\`)**

\- **\*\*Flexible Ingestion\*\***: Accepts both remote GitHub URLs (e.g., \`https\://github.com/pallets/flask\`) and local disk directories.

\- **\*\*Shallow Cloning & Caching\*\***: Clones repositories with \`--depth 1\` into \`data/repos/\` to minimize disk space and optimize ingestion speed.

\- **\*\*Strict File Filter\*\***: Automatically ignores lock files (\`package-lock.json\`, \`poetry.lock\`), binary files, virtual environments (\`.venv\`, \`node_modules\`, \`dist\`, \`build\`), and hidden directories (\`.git\`).



\---



**### 2. AST-Aware Code Parsing & Chunking (\`src/code_rag/core/parser.py\`, \`chunker.py\`)**

\- **\*\*Abstract Syntax Tree (AST)\*\***: Parses Python code into logical syntax nodes:

&#x20; \- **\*\*Classes\*\***: Retains class definitions, docstrings, and signatures.

&#x20; \- **\*\*Functions & Methods\*\***: Preserves complete functional bodies, arguments, and decorators.

&#x20; \- **\*\*Global & Module Scope\*\***: Captures top-level constants, configuration dictionaries, and imports.

\- **\*\*Exact Line Range Metadata\*\***: Every chunk tracks its \`start_line\`, \`end_line\`, \`file_path\`, \`symbol_name\`, and \`symbol_type\`.

\- **\*\*Repository Structure Chunk\*\***: Automatically creates an ASCII file-tree chunk (\`repository_file_tree\`) to enable repository overview queries.



\---



**### 3. Dual Hybrid Indexing (\`src/code_rag/storage/\`)**



To balance conceptual meaning with exact symbol matching, the project builds **\*\*two complementary indexes\*\***:



\`\`\`text

&#x20;                 AST-Parsed Code Chunks

&#x20;                           │

&#x20;           ┌───────────────┴───────────────┐

&#x20;           ↓                               ↓

&#x20;  [ Dense Vector Index ]         [ Sparse Lexical Index ]

&#x20;     ChromaDB + Embeddings              BM25Plus

&#x20;           │                               │

&#x20;  • Meaning-based search          • Exact keyword search

&#x20;  • Handles natural language      • Handles syntax & symbols

&#x20;  • Finds concepts (e.g. auth)    • Finds names (e.g. JWTAuth)

\`\`\`



1\. **\*\*Dense Vector Index (ChromaDB + \`nomic-embed-text\` / \`all-minilm\`)\*\***:

&#x20;  \- Converts each chunk into a mathematical vector representation.

&#x20;  \- **\*\*Semantic Understanding\*\***: If a user asks *\*"Where is user authentication handled?"\**, the vector store locates \`def verify_jwt_token()\` even if the word *\*"authenticated"\** is never explicitly used in that file.

2\. **\*\*Sparse Lexical Index (BM25Plus with Code Tokenizer)\*\***:

&#x20;  \- Uses an inverted index with code-aware tokenization (\`camelCase\`, \`snake_case\`, identifiers, variable names).

&#x20;  \- **\*\*Exact Symbol Matching\*\***: If a user queries for \`EngineGroupedStackingRegressor\` or \`calculate_loss()\`, BM25 matches the exact symbol with 100% precision without semantic drift.



\---



**### 4. Hybrid Reciprocal Rank Fusion (RRF) (\`src/code_rag/rag/retriever.py\`)**

Combines results from both the vector store and BM25 using Reciprocal Rank Fusion:



$$\text{RRF Score}(d) = \frac{1}{60 + \text{rank}\_{\text{vector}}(d)} + \frac{1}{60 + \text{rank}\_{\text{bm25}}(d)}$$



\- **\*\*Exact Path Boosting\*\***: If a user explicitly asks for a file (e.g., \`main.py\` or \`src/config.py\`), exact path matching prioritizes that complete file chunk with high confidence.

\- **\*\*Deduplication\*\***: Merges overlapping symbols and ranks the highest-quality candidate chunks for the LLM prompt.



\---



**### 5. Corrective LangGraph State Machine (\`src/code_rag/rag/graph.py\`)**

Queries run through a deterministic state graph:

1\. **\*\*Analyze Query\*\***: Detects user intent (\`symbol_lookup\`, \`file_retrieval\`, \`code_explanation\`, \`architecture_overview\`) and extracts symbols/file paths.

2\. **\*\*User-Pasted Code Extraction\*\***: If the user pastes raw code in the query and asks to explain it, the pipeline skips repository retrieval and routes directly to the **\*\*Line-by-Line Code Explainer\*\***.

3\. **\*\*Hybrid Retrieval\*\***: Fetches top-$K$ candidate chunks using RRF.

4\. **\*\*Relevance Grading\*\***: Verifies if retrieved code chunks answer the user's question. If irrelevant, triggers query rewriting to re-retrieve with alternate symbol names.

5\. **\*\*Grounded Generation\*\***: Feeds strictly bounded context to local Ollama LLMs with instructions to cite exact file paths, line ranges, and symbols.



\---



**### 6. Real-Time Streaming & ChatGPT-Style UI (\`src/code_rag/static/\`)**

\- **\*\*Server-Sent Events (SSE)\*\***: Streams tokens in real time directly from the server to the browser (\`/api/query/stream\`).

\- **\*\*Instant Response (<0.5s)\*\***: Complete source file requests stream immediately with a smooth line-by-line typing animation followed by the AI explanation.

\- **\*\*ChatGPT-Style Code Block Header\*\***:

&#x20; \- **\*\*Language Tag\*\***: Displays detected language (\`python\`, \`typescript\`, \`cpp\`, \`html\`, etc.).

&#x20; \- **\*\*Copy Button\*\***: One-click full code copying with checkmark feedback.

&#x20; \- **\*\*Download Button\*\***: One-click download with matching file extension (\`.py\`, \`.js\`, \`.ts\`, \`.css\`, \`.html\`, \`.cpp\`, \`.c\`, \`.rs\`, \`.go\`, etc.) and matching filename.

\- **\*\*Interactive Repository Management\*\***:

&#x20; \- Delete any specific indexed repository with one click.

&#x20; \- Delete button stays hidden by default and smoothly reveals on hover.

\- **\*\*Dynamic Navbar \`+\` Button\*\***:

&#x20; \- When the sidebar is collapsed or hidden, a dedicated \`+ New Chat\` button automatically appears in the top navigation bar (with \`Ctrl+N\` / \`Cmd+N\` shortcut).



\---



**## ⚡ Key Features**



\| Feature | Description |

\| :--- | :--- |

\| **\*\*AST Code Parsing\*\*** | Parses Python AST nodes (classes, methods, functions) with exact line bounds. |

\| **\*\*Hybrid RRF Search\*\*** | Dense embeddings (ChromaDB) + Sparse lexical index (BM25Plus) for high retrieval precision. |

\| **\*\*Corrective LangGraph\*\*** | Self-correcting state machine that grades relevance and rewrites queries when needed. |

\| **\*\*ChatGPT-Style Streaming\*\*** | Instant startup (<0.5s) with real-time line-by-line code typing animation. |

\| **\*\*One-Click Copy & Download\*\*** | Download any code block with the correct file extension and exact name. |

\| **\*\*Line-by-Line Explainer\*\*** | Paste any code snippet and get an in-depth breakdown of every line. |

\| **\*\*Hover-to-Delete Repos\*\*** | Easily remove specific indexed repositories from vector and keyword storage. |

\| **\*\*Local & Private\*\*** | Runs 100% locally with Ollama (\`llama3.2\`, \`olmo\`) — zero API keys or external data leaks. |



\---



**## 🚀 Quickstart & Installation**



**### 1. Prerequisites**

\- **\*\*Python 3.9+\*\***

\- **\*\*[Ollama]\(**&#x68;ttps\://ollama.co&#x6D;**)\*\*** installed and running:



\`\`\`bash

\# Pull the embedding model

ollama pull nomic-embed-text



\# Pull the LLM model

ollama pull llama3.2:latest

\`\`\`



**### 2. Clone & Install Dependencies**



\`\`\`bash

git clone https\://github.com/ashikhasanredoy/Github-repositories-read-RAG.git

cd Github-repositories-read-RAG



\# Create and activate virtual environment

python3 -m venv venv

source venv/bin/activate



\# Install dependencies

pip install -r requirements.txt

\`\`\`



**### 3. Launch the Server**



\`\`\`bash

python3 run.py

\`\`\`



Open your browser at **\*\*[http\://localhost:8000]\(**&#x68;ttp\://localhost:800&#x30;**)\*\***.



\---



**## 🖥️ User Guide & Examples**



**### Ingesting a Codebase**

1\. Enter a GitHub URL (e.g. \`https\://github.com/pallets/flask\`) or a local directory path in the sidebar.

2\. Click **\*\*Index Repository\*\***.

3\. The system parses AST structures, computes embeddings, and builds the BM25 index.



**### Example Prompts to Try:**

\- **\*\*Full File Retrieval\*\***: \`"give me main.py file"\`

\- **\*\*Line-by-Line Explanation\*\***:

&#x20; \`\`\`text

&#x20; explain this code line by line:

&#x20; def predict(features):

&#x20;     scaled = scaler.transform(features)

&#x20;     return model.predict(scaled)

&#x20; \`\`\`

\- **\*\*Architecture & Workflow\*\***: \`"Explain the data preprocessing and model evaluation workflow."\`

\- **\*\*Symbol Lookup\*\***: \`"Where is the Voting Ensemble created and what models are included?"\`



\---



**## 🔌 REST API Reference**



\| Endpoint | Method | Description |

\| :--- | :--- | :--- |

\| \`GET /api/health\` | \`GET\` | Health check & Ollama connection status |

\| \`GET /api/models\` | \`GET\` | List available local Ollama LLMs |

\| \`GET /api/repos\` | \`GET\` | List all indexed repository collections |

\| \`POST /api/index\` | \`POST\` | Clones, parses, and indexes a repository |

\| \`POST /api/query\` | \`POST\` | Executes standard RAG pipeline (returns JSON answer + citations) |

\| \`POST /api/query/stream\` | \`POST\` | SSE endpoint for real-time token-by-token streaming |

\| \`DELETE /api/repos/{repo_id}\` | \`DELETE\` | Deletes a repository collection from ChromaDB, BM25, and disk |



\---



**## 📁 Repository Structure**



\`\`\`

Github-repositories-read-RAG/

├── assets/

│   └── pipeline_diagram.jpg     # Architecture diagram

├── data/

│   ├── chroma_db/               # Persistent ChromaDB vector database

│   ├── bm25/                    # Serialized BM25Plus keyword indices

│   └── repos/                   # Cloned repository working copies

├── src/

│   └── code_rag/

│       ├── config.py            # System configuration & file ignore rules

│       ├── api/

│       │   ├── main.py          # FastAPI application & REST/SSE endpoints

│       │   └── schemas.py       # Pydantic request/response schemas

│       ├── core/

│       │   ├── cloner.py        # Shallow Git cloner & local path resolver

│       │   ├── parser.py        # AST symbol extractor & line bound parser

│       │   ├── chunker.py       # Code chunker with syntax boundaries

│       │   └── models.py        # CodeChunk, RepoMetadata data models

│       ├── storage/

│       │   ├── vector_store.py  # ChromaDB dense vector store manager

│       │   └── bm25_store.py    # BM25Plus sparse indexer & code tokenizer

│       ├── rag/

│       │   ├── retriever.py     # Hybrid RRF fusion retriever & context builder

│       │   ├── state.py         # LangGraph TypedDict state

│       │   └── graph.py         # LangGraph workflow & real-time streaming engine

│       ├── services/

│       │   └── ollama.py        # Ollama client with dynamic context scaling

│       └── static/

│           ├── index.html       # Web application UI

│           ├── styles.css       # Modern dark/light theme design system

│           └── app.js           # Frontend client logic & event listeners

├── requirements.txt             # Python project dependencies

├── run.py                       # Application launcher

└── README.md                    # Project documentation

\`\`\`



\---



**## 🎨 Acknowledgements & Credits**



\- **\*\*Frontend Design & Development\*\***: The modern, responsive ChatGPT-inspired web user interface, glassmorphism design system, real-time streaming visual effects, dynamic code block headers (Copy & Download buttons), and interactive repository management were designed and developed with assistance from **\*\*Claude (Anthropic)\*\***.

\- **\*\*Backend & RAG Architecture\*\***: Designed and developed by **\*\*[Ashik Hasan Redoy]\(**&#x68;ttps\://github.com/ashikhasanredo&#x79;**)\*\*** using FastAPI, LangGraph, ChromaDB, BM25Plus, AST syntax parsing, and Ollama.



\---



**## 📄 License**

MIT © 2026 Ashik Hasan Redoy
