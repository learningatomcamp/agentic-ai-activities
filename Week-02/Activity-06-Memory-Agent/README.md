# Week 2 · Activity 02: Personal AI Assistant

**Lesson 6: Agent Memory**

Build a personal AI assistant that remembers you. It keeps track of the current conversation, saves useful facts and preferences about you to a file, and uses them to personalise its answers, even in a brand-new session.

This activity builds on the **Lesson 5 RAG activity**. There, the agent retrieved information from a knowledge base. Here, it retrieves information about **you**, and it also decides what new information is worth saving.

## What you'll learn

- Why agents need memory, and why LLMs don't have it built in
- The difference between short-term memory (the conversation) and long-term memory (saved facts)
- How to keep and limit a conversation history
- What an assistant should remember, and what it must never store
- How to store, retrieve and use memories with a simple JSON file

## What's in this folder

| File | What it is |
|---|---|
| `Lesson_06_Activity.ipynb` | The guided notebook. It explains each idea and builds the full assistant step by step. **Start here.** |
| `lesson_06_memory_agent.py` | A partially completed starter file. You complete six `TODO` sections to build the same assistant yourself. |
| `README.md` | This file |

When you run the code, it also creates:

| File | What it is |
|---|---|
| `memory.json` | The assistant's long-term memory. Open it any time to see exactly what's remembered. |
| `memory_test_results.json` | The results of the two-session memory test. You submit this file. |

## Tools used

| Tool | Purpose |
|---|---|
| Python 3.10+ | The programming language |
| Gemini API (`google-genai`) | The assistant's reasoning model, the same as Lesson 5 |
| `json` (built into Python) | Stores long-term memory in a readable file |

No database or extra frameworks are needed.

---

## Setup

### Option A: Google Colab (easiest)

1. Open `Lesson_06_Activity.ipynb` in [Google Colab](https://colab.research.google.com) (**File → Upload notebook**, or open it from GitHub).
2. Run the first code cell. It installs everything you need.
3. That's it. The API key is entered inside the notebook (see below).

`memory.json` is saved in Colab's **Files** panel (left sidebar). It survives while the runtime is running, but it's deleted when the runtime is reset. Download it if you want to keep it.

To run the `.py` file in Colab, upload it to the Files panel and run it in a code cell:

```python
!pip -q install google-genai
import os, getpass
os.environ["GEMINI_API_KEY"] = getpass.getpass("Enter your Gemini API key: ")
!python lesson_06_memory_agent.py --test
```

The `--chat` mode needs a terminal. In Colab, use the chat cell in the notebook instead.

### Option B: Your own computer

You need **Python 3.10 or newer**. Check with:

```bash
python --version
```

Create a virtual environment and install the packages:

```bash
# Go to this activity's folder
cd Week-02/Activity-02-Personal-AI-Assistant

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

# Install the packages
pip install google-genai jupyter
```

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
jupyter notebook Lesson_06_Activity.ipynb
```

The notebook has 11 sections:

| # | Section | # | Section |
|---|---|---|---|
| 1 | Why agents need memory | 7 | Retrieving relevant memories |
| 2 | Short-term vs long-term memory | 8 | Giving retrieved memories to the LLM |
| 3 | Setup | 9 | Building the Personal AI Assistant |
| 4 | Conversation history | 10 | Testing memory across multiple interactions |
| 5 | User preferences and facts | 11 | Challenge |
| 6 | Storing memories | | |

---

## How to complete and run the `.py` file

After the notebook, open `lesson_06_memory_agent.py`. Most of it is written for you. Complete the six `TODO` sections:

| TODO | Where | What you do |
|---|---|---|
| 1 | `load_memories()` | Read the memories from `memory.json` |
| 2 | `save_memories()` | Write the memories to `memory.json` |
| 3 | `save_memory()` | Build a new memory record and store it. This is the tool the assistant calls. |
| 4 | `retrieve_memories()` | Score each memory by how many words it shares with the message |
| 5 | `format_memories()` | Turn memories into text lines the LLM can read |
| 6 | `chat()` | Connect memory to the assistant: retrieve, add to the system instruction, and update the conversation history |

If you run the file before finishing a `TODO`, it stops with a message naming the `TODO` you still need to complete. The messages may not appear in number order (for example, `--seed` reaches TODO 3 before TODO 1), so complete whichever one is named.

Run every command from inside the activity folder.

**Step 1: Test storage and retrieval (TODOs 1–5).** No API key is needed for these commands.

```bash
python lesson_06_memory_agent.py --seed                       # add 3 sample memories about "Sam"
python lesson_06_memory_agent.py --show                       # print all saved memories
python lesson_06_memory_agent.py --find "any snack ideas?"    # test retrieval
```

`--find "any snack ideas?"` should return the peanut-allergy memory, plus Sam's name (profile memories are always included). Open `memory.json` to see how the memories are stored.

**Step 2: Run the memory test (all six TODOs).**

```bash
python lesson_06_memory_agent.py --test
```

This deletes the sample memories and runs two sessions:

- **Session 1:** Amara introduces herself, shares her goal and preferences, and asks one off-topic question.
- **Session 2:** a new session with short-term memory cleared. The assistant has to remember Amara using only `memory.json`.

It runs 8 checks and saves the results to **`memory_test_results.json`**.

| Check | What it tests |
|---|---|
| Name, vegetarian, data analyst saved | The assistant decided to save the right things |
| Houseplant question not saved | The assistant didn't save a one-off question |
| Name remembered in session 2 | Long-term memory survives a new session |
| Dinner and career questions retrieved the right memory | Retrieval found relevant memories |
| Capital-of-Japan question saved nothing | No unnecessary memories |

The model can behave a little differently on each run. If a check shows `CHECK`, read the `[memory] saved:` and `[memory] retrieved:` lines in the output to understand why.

**Step 3: Chat with your assistant.**

```bash
python lesson_06_memory_agent.py --chat
```

Type `new` to start a new session, `memories` to see what's saved, or `quit` to exit. Your memories stay in `memory.json` until you run:

```bash
python lesson_06_memory_agent.py --reset
```

---

## Challenge

The challenge is in **Section 11** of the notebook.

**Part A: Let the assistant forget.** Complete a `forget_memory` tool so the assistant can remove memories that are no longer true. Test it with *"I'm not vegetarian anymore, I eat fish now."*

**Part B: Test the privacy rule.** Ask the assistant to remember a bank PIN, then check `memory.json`. Add a code-level guard to `save_memory` that refuses secrets, and explain why a code guard is safer than an instruction alone.

**Stretch:** replace keyword matching with the embeddings from Lesson 5.

---

## What to submit

Submit these through the activity's submission form on the course website:

1. **`Lesson_06_Activity.ipynb`**: run all cells, including the challenge, so the outputs are saved in the notebook. In Colab, use **File → Download → Download .ipynb**. Make sure no API key appears anywhere in it.
2. **`lesson_06_memory_agent.py`**: with all six TODOs completed.
3. **`memory_test_results.json`**: created by running `python lesson_06_memory_agent.py --test`.
4. **A short reflection (100–150 words)** answering:
   - Which memories did the assistant save in session 1? Were they the right ones?
   - In session 2, how did the assistant answer "Suggest a quick dinner idea" without being told you're vegetarian?
   - In Part B, did the assistant refuse to save the PIN? Why is a code-level guard safer than an instruction?

### Checklist before you submit

- [ ] All six TODOs in the `.py` file are complete
- [ ] `python lesson_06_memory_agent.py --test` runs without errors
- [ ] `memory_test_results.json` shows at least 6 of 8 checks passing
- [ ] Both challenge parts are completed in the notebook, with outputs visible
- [ ] Your API key does not appear in any file
- [ ] `memory.json` contains no real personal information. Use test details like the ones in the activity.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `GEMINI_API_KEY is not set` | Set the environment variable in the **same** terminal you run the file from (see API key setup). |
| `NotImplementedError: TODO ...` | You haven't finished that TODO yet. Complete it and run the file again. |
| `--show` says "No memories saved yet" | Run `--seed` first, or chat with the assistant to create memories. |
| `--find` returns only the name | Check TODO 4. Make sure you include the memory's **keywords** and **category** when scoring, not just its text. |
| `JSONDecodeError` when loading memories | `memory.json` was edited by hand and has a mistake. Fix it, or run `--reset` to start again. |
| The assistant forgets your name in a new session | Check that the name was saved (`--show`) and that TODO 6a/6b pass memories into the system instruction. |
| The assistant saves everything, including questions | The base instructions say not to. Check that TODO 6b uses `build_system_instruction()`. |
| `503 UNAVAILABLE` or `429 RESOURCE_EXHAUSTED` | Gemini is busy or you hit the rate limit. The code retries automatically; if it keeps failing, wait a minute. |
| `404` model not found | Set `GEMINI_MODEL` to a current model name from Google AI Studio. |
