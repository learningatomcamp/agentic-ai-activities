# Week 2 · Activity 03: LangGraph Agent

**Lesson 8: LangChain & LangGraph**

Build a **Course Helpdesk agent** as a LangGraph workflow. It reads each request, decides what kind it is, and sends it down the right path:

| Request | Path |
|---|---|
| A question about the course | Search the knowledge base → answer with sources |
| A calculation | Calculator agent ⇄ calculator tool, looping until done |
| "Send a message to…" | Draft the message → **pause for human approval** → send or cancel |
| Greetings and small talk | Reply directly |

It also remembers each conversation thread, so follow-up questions work.

In Lessons 5 and 6 you built retrieval, tools and memory by hand with the Gemini SDK. This activity shows how **LangChain** packages those pieces as ready-made components, and how **LangGraph** connects them into a workflow you control.

## What you'll learn

- What LangChain is, and how its models, prompts, tools, retrievers and agents fit together
- What LangGraph adds: state, nodes, edges and conditional routing
- How to build a tool loop inside a graph
- How checkpointing gives an agent persistent conversations
- How to pause a workflow for human approval with `interrupt()`

## What's in this folder

| File | What it is |
|---|---|
| `Lesson_08_Activity.ipynb` | The guided notebook. It explains each idea and builds the full helpdesk step by step. **Start here.** |
| `lesson_08_langgraph_agent.py` | A partially completed starter file. You complete seven `TODO` sections to build the same agent yourself. |
| `README.md` | This file |

Running the tests creates `langgraph_test_results.json`, which you submit.

## Tools used

| Tool | Purpose |
|---|---|
| Python 3.10+ | The programming language |
| `langchain` | Models, prompts, tools, retrievers and prebuilt agents |
| `langchain-google-genai` | Connects LangChain to Gemini (`gemini-3.8-flash`) and Gemini embeddings (`gemini-embedding-001`) |
| `langgraph` | State, nodes, edges, routing, checkpointing and human-in-the-loop |

All three packages use the same Gemini API key as the previous labs. The knowledge base is the same 8 documents from Lesson 5.

---

## Setup

### Option A: Google Colab (easiest)

1. Open `Lesson_08_Activity.ipynb` in [Google Colab](https://colab.research.google.com) (**File → Upload notebook**, or open it from GitHub).
2. Run the first code cell. It installs everything you need.
3. That's it. The API key is entered inside the notebook (see below).

To run the `.py` file in Colab, upload it to the **Files** panel and run it in a code cell:

```python
!pip -q install langchain langgraph langchain-google-genai
import os, getpass
os.environ["GEMINI_API_KEY"] = getpass.getpass("Enter your Gemini API key: ")
!python lesson_08_langgraph_agent.py --test
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
cd Week-02/Activity-03-LangGraph-Agent

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

# Install the packages
pip install langchain langgraph langchain-google-genai jupyter
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

LangChain's Gemini integration reads `GEMINI_API_KEY` automatically. (`GOOGLE_API_KEY` also works.)

**Optional:** to use a different Gemini model, set `GEMINI_MODEL` the same way. The default is `gemini-3.8-flash`.

---

## How to run the notebook

**In Colab:** open the notebook and run the cells from top to bottom with **Shift + Enter**.

**On your computer:**

```bash
jupyter notebook Lesson_08_Activity.ipynb
```

The notebook has 14 sections:

| # | Section | # | Section |
|---|---|---|---|
| 1 | What is LangChain, and why is it useful? | 8 | Conditional routing |
| 2 | Models and prompts | 9 | Build a simple LangGraph agent |
| 3 | Tools and retrievers | 10 | Add a tool and route based on the request |
| 4 | What LangGraph adds to agent workflows | 11 | Add basic persistence (checkpointing) |
| 5 | State | 12 | A simple human-in-the-loop step |
| 6 | Nodes | 13 | Test the workflow |
| 7 | Edges | 14 | Challenge |

The graph is built in three versions, so you can see it grow: questions and chat (Section 9), then calculations (Section 10), then the approval path (Section 12).

---

## How to complete and run the `.py` file

After the notebook, open `lesson_08_langgraph_agent.py`. The LangChain components (model, retriever, tool), the prompts, and the `answer`, `calculator_agent`, `chat` and `draft_action` nodes are written for you. Complete the seven `TODO` sections:

| TODO | Where | What you do |
|---|---|---|
| 1 | `HelpdeskState` | **State:** add the five fields the graph passes between nodes |
| 2 | `classify()` | **Node:** ask the router for a decision and return the route |
| 3 | `retrieve()` | **Node:** search the knowledge base and return the context and sources |
| 4 | `human_approval()` | **Node:** pause the graph with `interrupt()` and wait for a human decision |
| 5 | `route_request()` | **Routing:** map the route to the name of the next node |
| 6 | `build_graph()` | **Edges:** add the conditional edge from `classify`, the calculator loop, and the approval and chat paths |
| 7 | `build_graph()` | **Workflow:** compile the graph with a checkpointer |

If you run the file before finishing a `TODO`, it stops with a message naming the `TODO` you still need to complete. When the error happens inside a node, LangGraph adds a line like `During task with name 'retrieve'` after it. The TODO message is the line just above.

Run every command from inside the activity folder.

**Step 1: Check the graph structure (TODOs 1, 6 and 7).** No API key is needed.

```bash
python lesson_08_langgraph_agent.py --graph
```

This prints the graph as a Mermaid diagram. Paste it into [mermaid.live](https://mermaid.live) to see it drawn. Check that every path reaches `END`.

**Step 2: Run the tests (all seven TODOs).**

```bash
python lesson_08_langgraph_agent.py --test
```

This sends five test requests, each in its own thread. It answers the approval requests automatically (one "yes", one "no"), then runs a two-message conversation to check persistence. It runs 11 checks and saves **`langgraph_test_results.json`**.

| Test | What it checks |
|---|---|
| "What is prompt injection?" | Routed to `question`; retrieved **Agent safety and guardrails** |
| "What is 15% of 2,480, plus 99?" | Routed to `calculation`; used the `calculate` tool; answer is **471** |
| "Hi! What can you help me with?" | Routed to `chat` |
| "Send a reminder to my study group…" | Routed to `action`; **sent** after the human said yes |
| "Email my instructor…" | Routed to `action`; **cancelled** after the human said no |
| "What is RAG?" then "Explain that…" | The checkpointer kept both turns in the same thread |

The router is an LLM, so it can occasionally choose a different path. If a check shows `CHECK`, read the `[classify]` line in the output: its reason usually explains the choice.

**Step 3: Chat with your agent.**

```bash
python lesson_08_langgraph_agent.py --chat
```

When the agent wants to send a message, it shows you the draft and asks you to approve it. Type `new` to start a new conversation thread, or `quit` to exit. Messages are never really sent; they go to a pretend outbox.

---

## Challenge

The challenge is in **Section 14** of the notebook.

**Part A: Add a new route.** Add a **quiz** path that retrieves a document and writes one multiple-choice question about it. Update the router, the routing function and the graph, and print the new diagram.

**Part B: Let the human edit the draft.** Change `human_approval` so a person can approve, cancel, or reply `edit: ...` to send a corrected message instead.

**Stretch:** replace `InMemorySaver` with `SqliteSaver` (`pip install langgraph-checkpoint-sqlite`), so conversations survive a restart.

---

## What to submit

Submit these through the activity's submission form on the course website:

1. **`Lesson_08_Activity.ipynb`**: run all cells, including the challenge, so the outputs are saved in the notebook. In Colab, use **File → Download → Download .ipynb**. Make sure no API key appears anywhere in it.
2. **`lesson_08_langgraph_agent.py`**: with all seven TODOs completed.
3. **`langgraph_test_results.json`**: created by running `python lesson_08_langgraph_agent.py --test`.
4. **Your graph diagram**: a screenshot from [mermaid.live](https://mermaid.live) of your final graph, including your challenge route.
5. **A short reflection (100–150 words)** answering:
   - Compare building the agent with LangGraph to building it by hand in Lessons 5 and 6. What got easier? What became harder to see?
   - Why does `human_approval` need a checkpointer to work?
   - Which request types did your router get wrong, if any, and how could you improve it?

### Checklist before you submit

- [ ] All seven TODOs in the `.py` file are complete
- [ ] `python lesson_08_langgraph_agent.py --graph` prints a diagram where every path reaches `END`
- [ ] `python lesson_08_langgraph_agent.py --test` runs without errors
- [ ] `langgraph_test_results.json` shows at least 9 of 11 checks passing
- [ ] Both challenge parts are completed in the notebook, with outputs visible
- [ ] Your API key does not appear in any file

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `GEMINI_API_KEY is not set` | Set the environment variable in the **same** terminal you run the file from (see API key setup). |
| `NotImplementedError: TODO ...` | You haven't finished that TODO yet. Complete it and run the file again. |
| `TODO 6: node '...' has no outgoing edge` | Every node needs an edge leaving it, either to another node or to `END`. |
| A node's update seems to disappear, or you get a `KeyError` on a state field | The field isn't declared in `HelpdeskState`, so LangGraph ignores it. Check TODO 1. |
| `Checkpointer requires one or more of the following 'configurable' keys` | You called `.invoke()` without a thread id. Pass `{"configurable": {"thread_id": "..."}}` as the second argument. |
| `Cannot use Command(resume=...) without checkpointer` | Human-in-the-loop needs a checkpointer to save the paused state. Check TODO 7. |
| The calculator loop never ends | `tools_condition` must be the conditional edge **from** `calculator_agent`, and `tools` must lead back to it. LangGraph stops after 25 steps (`GraphRecursionError`). |
| A follow-up question doesn't remember the previous message | Use the **same** thread id for both messages. |
| `503 UNAVAILABLE` or `429 RESOURCE_EXHAUSTED` | Gemini is busy or you hit the rate limit. Wait a minute and try again. |
| `404` model not found | Set `GEMINI_MODEL` to a current model name from Google AI Studio. |
