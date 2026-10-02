# Week 2 · Lab 05: Agent Knowledge Base

**Lesson 5: RAG for Agents**

Build a knowledge-aware AI agent. It searches a small knowledge base when a question needs it, answers using what it found, names its sources, and says so honestly when the knowledge base doesn't cover a question.

This activity builds directly on the **Week 1 tool-calling activity**. The knowledge base becomes one more tool, and the agent decides when to use it.

## What you'll learn

- Why agents need external knowledge
- What RAG (Retrieval-Augmented Generation) is: retrieve, augment, generate
- How embeddings and semantic search work
- How to store and search documents with ChromaDB
- How to turn retrieval into a tool that Gemini calls only when it's needed

## What's in this folder

| File | What it is |
|---|---|
| `Lesson_05_Activity.ipynb` | The guided notebook. It explains each idea and builds the full agent step by step. **Start here.** |
| `lesson_05_rag_agent.py` | A partially completed starter file. You complete five `TODO` sections to build the same agent yourself. |
| `README.md` | This file |

## Tools used

| Tool | Purpose |
|---|---|
| Python 3.10+ | The programming language |
| Gemini API (`google-genai`) | The agent's reasoning model, the same as Week 1 |
| `sentence-transformers` | Creates embeddings for free, on your own machine (model: `all-MiniLM-L6-v2`) |
| ChromaDB | Stores the embeddings and searches them by meaning |

No other frameworks are needed.

---

## Setup

### Option A: Google Colab (easiest)

1. Open `Lesson_05_Activity.ipynb` in [Google Colab](https://colab.research.google.com) (**File → Upload notebook**, or open it from GitHub).
2. Run the first code cell. It installs everything you need.
3. That's it. The API key is entered inside the notebook (see below).

To run the `.py` file in Colab, upload it to the Files panel (left sidebar) and run it in a code cell:

```python
!pip -q install google-genai sentence-transformers chromadb
import os, getpass
os.environ["GEMINI_API_KEY"] = getpass.getpass("Enter your Gemini API key: ")
!python lesson_05_rag_agent.py
```

### Option B: Your own computer

You need **Python 3.10 or newer**. Check with:

```bash
python --version
```

Create a virtual environment and install the packages:

```bash
# Go to this activity's folder
cd Week-02/Activity-01-Agent-Knowledge-Base

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

# Install the packages
pip install google-genai sentence-transformers chromadb jupyter
```

The first time you run the code, `sentence-transformers` downloads the embedding model (about 90 MB). After that, it loads from your computer.

---

## API key setup

Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com): sign in and choose **Get API key**.

**Never paste your key into a code cell or a `.py` file**, and never commit it to GitHub.

### For the notebook

When you run the API key cell, a box appears. Paste your key there. It's hidden as you type and isn't saved in the notebook.

### For the `.py` file

Set the key as an environment variable in the same terminal you'll run the file from:

```bash
# macOS / Linux
export GEMINI_API_KEY="paste-your-key-here"

# Windows (Command Prompt)
set GEMINI_API_KEY=paste-your-key-here

# Windows (PowerShell)
$env:GEMINI_API_KEY="paste-your-key-here"
```

You'll need to set it again each time you open a new terminal.

**Optional:** to use a different Gemini model, set `GEMINI_MODEL` the same way. The default is `gemini-3.8-flash`.

---

## How to run the notebook

**In Colab:** open the notebook and run the cells from top to bottom with **Shift + Enter**.

**On your computer:**

```bash
jupyter notebook Lesson_05_Activity.ipynb
```

The notebook has 13 sections:

| # | Section | # | Section |
|---|---|---|---|
| 1 | Why agents need external knowledge | 8 | Create the `knowledge_search()` tool |
| 2 | What is RAG? | 9 | Give the tool to Gemini |
| 3 | Setup | 10 | Let the agent decide when to retrieve |
| 4 | Embeddings and semantic search | 11 | Return retrieved information and generate the final answer |
| 5 | Create the knowledge base | 12 | Test the agent |
| 6 | Generate embeddings | 13 | Challenge |
| 7 | Store and search embeddings with ChromaDB | | |

---

## How to complete and run the `.py` file

After the notebook, open `lesson_05_rag_agent.py`. Most of it is written for you. Complete the five `TODO` sections **in order**:

| TODO | Where | What you do |
|---|---|---|
| 1 | `embed()` | Turn a list of texts into embeddings with the embedding model |
| 2 | `build_knowledge_base()` | Create embeddings for the documents and add them to ChromaDB |
| 3 | `knowledge_search()` | Embed the query, search ChromaDB, and skip weak matches |
| 4 | `knowledge_search_declaration` | Write the tool description that tells Gemini **when** to search |
| 5 | `run_rag_agent()` | Run the tool Gemini asks for and send the result back to it |

If you run the file before finishing a `TODO`, it stops with a message naming the `TODO` you still need to complete.

Run every command from inside the activity folder.

**Step 1: Test retrieval only.** Once TODOs 1–3 are done, test the search. No API key is needed for this step.

```bash
python lesson_05_rag_agent.py --search "stopping rules for agent loops"
```

You should see **The ReAct loop and stopping rules** as the top result, with a similarity score.

**Step 2: Ask one question.** Once all five TODOs are done:

```bash
python lesson_05_rag_agent.py --ask "What is prompt injection?"
```

You should see the agent search, find **Agent safety and guardrails**, and answer with that title listed as a source.

**Step 3: Run all the tests.**

```bash
python lesson_05_rag_agent.py
```

This runs six test questions, checks whether the agent made the right retrieval decision for each, and saves the results to **`rag_test_results.json`**.

| Question type | What should happen |
|---|---|
| Covered by the knowledge base | The agent searches, then answers with sources |
| Greetings and simple maths | The agent answers directly without searching |
| About agents, but not in the knowledge base | The agent searches, finds nothing useful, and says so |

The model can make a different decision on different runs. If a decision shows `CHECK THIS`, read the agent's search query in the output to understand why.

---

## Challenge

The challenge is in **Section 13** of the notebook.

**Part A: Grow the knowledge base.** Write two new documents on topics the knowledge base doesn't cover (for example, *Evaluating agents* or *Planning in agents*). Rebuild the knowledge base, update the tool description, and test it with three questions of your own:

- one answered by a new document,
- one that needs a new document and an original one,
- one the knowledge base still can't answer.

**Part B: Tune the threshold.** Run the same question with `MIN_SIMILARITY` set to `0.1`, `0.35` and `0.6`. Note how the search results and answers change.

**Stretch:** add a `topic` to each document's metadata, and let `knowledge_search` filter by topic using ChromaDB's `where` argument.

---

## What to submit

Submit these through the activity's submission form on the course website:

1. **`Lesson_05_Activity.ipynb`**: run all cells, including the challenge, so the outputs are saved in the notebook. In Colab, use **File → Download → Download .ipynb**. Make sure no API key appears anywhere in it.
2. **`lesson_05_rag_agent.py`**: with all five TODOs completed.
3. **`rag_test_results.json`**: created by running `python lesson_05_rag_agent.py`.
4. **A short reflection (100–150 words)** answering:
   - Find one question where the agent **didn't** search. Was that the right decision? Why?
   - In Part B, which `MIN_SIMILARITY` value would you choose, and why?
   - How did the agent handle the question the knowledge base couldn't answer?

### Checklist before you submit

- [ ] All five TODOs in the `.py` file are complete
- [ ] `python lesson_05_rag_agent.py` runs without errors
- [ ] `rag_test_results.json` shows at least 5 of 6 retrieval decisions as expected
- [ ] Both challenge parts are completed in the notebook, with outputs visible
- [ ] Your API key does not appear in any file

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `GEMINI_API_KEY is not set` | Set the environment variable in the **same** terminal you run the file from (see API key setup). |
| `NotImplementedError: TODO ...` | You haven't finished that TODO yet. Complete it and run the file again. |
| The first run is slow | The embedding model is downloading (about 90 MB). It's only slow the first time. |
| `--search` returns no results for a relevant query | Check TODO 3c. You may be skipping documents that are **above** the threshold instead of below it. |
| The agent never searches | Your TODO 4 description is too vague. Say what topics the knowledge base covers and when to use it. |
| The agent searches for greetings or maths | Your TODO 4 description should say when **not** to use the tool. |
| `503 UNAVAILABLE` or `429 RESOURCE_EXHAUSTED` | Gemini is busy or you hit the rate limit. The code retries automatically; if it keeps failing, wait a minute. |
| `404` model not found | Set `GEMINI_MODEL` to a current model name from Google AI Studio. |
| ChromaDB error mentioning `sqlite3` | Your Python's SQLite version is too old. Use Colab, or upgrade Python. |
| "Extra inputs are not permitted" error mentioning `type` | Gemini tool declarations don't use that key. Remove it (see notebook Section 9). |
