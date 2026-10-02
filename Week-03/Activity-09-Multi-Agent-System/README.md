# Week 2 · Lab 09: Research Team Agent

**Lesson 9: Multi-Agent Systems**

Build a small **research team** of AI agents that collaborate on one research question. Four specialists each do one job, and a supervisor coordinates them:

| Team member | Job | Tools / output |
|---|---|---|
| **Researcher** | Gathers facts with sources | Tools: `search_course_notes`, `search_wikipedia` |
| **Analyst** | Organises the facts into key points, gaps and an outline | Structured output |
| **Writer** | Writes a short report, and revises it after feedback | Prompt + model chain |
| **Reviewer** | Checks the report against four criteria; approves it or sends it back | Structured output |
| **Supervisor** | Decides who works next, and when the team is done | Rules in plain Python |

This activity builds on the **Lesson 8 LangGraph activity**. There you built one agent with several paths. Here, several agents share one workflow.

## What you'll learn

- When multiple agents are better than one, and when they aren't
- How to design specialised agents and delegate work between them
- The supervisor (hub-and-spoke) architecture
- How agents communicate through shared state
- How to build a feedback loop with a stopping rule

## What's in this folder

| File | What it is |
|---|---|
| `Lesson_09_Activity.ipynb` | The guided notebook. It explains each idea and builds the full team step by step. **Start here.** |
| `lesson_09_multi_agent.py` | A partially completed starter file. You complete seven `TODO` sections to build the same team yourself. |
| `README.md` | This file |

Running the tests creates `research_team_results.json` and `research_report.md`, which you submit.

## Tools used

| Tool | Purpose |
|---|---|
| Python 3.10+ | The programming language |
| `langchain` | The Researcher agent (`create_agent`), tools, prompts and the retriever |
| `langchain-google-genai` | Connects LangChain to Gemini (`gemini-3.8-flash`) and Gemini embeddings (`gemini-embedding-001`) |
| `langgraph` | Shared state, nodes, edges and the supervisor workflow |
| `requests` | The Wikipedia search tool (no key needed) |

These are the same packages as Lesson 8, and the same Gemini API key. The course notes are the 8 documents from Lessons 5 and 8.

---

## How the workflow works

The team uses the **supervisor architecture**. The supervisor sits in the middle, and every agent reports back to it after doing its job:

```text
                         START
                           ↓
            ┌──────────► SUPERVISOR ◄──────────┐
            │      ┌────────┼────────┬─────────┤
            │      ↓        ↓        ↓         ↓
            │  Researcher Analyst  Writer  Reviewer
            │      │        │        │         │
            └──────┴────────┴────────┴─────────┘
                           ↓
                        finish → END
```

The agents never talk to each other directly. They communicate through a **shared state**: each agent reads what it needs and writes only its own fields.

| Agent | Reads | Writes |
|---|---|---|
| Researcher | `topic` | `notes` |
| Analyst | `topic`, `notes` | `analysis` |
| Writer | `topic`, `notes`, `analysis`, `review` | `draft`, `revision` |
| Reviewer | `topic`, `notes`, `draft` | `review` |
| Supervisor | everything | `next_agent` |
| finish | `draft`, `review` | `final_report` |

Every agent also adds one line to `team_log`, so you can read the story of the collaboration.

The supervisor checks these rules in order:

| Rule | Next |
|---|---|
| No notes yet | Researcher |
| No analysis yet | Analyst |
| No draft yet | Writer |
| The latest draft hasn't been reviewed | Reviewer |
| The review approved it | finish |
| Rejected, and revisions are still allowed (`MAX_REVISIONS = 2`) | Writer, to revise using the feedback |
| Rejected, and the revision limit is reached | finish (the report says it wasn't approved) |

The revision limit is the team's **stopping rule**, like the step limit from Lesson 4.

---

## Setup

### Option A: Google Colab (easiest)

1. Open `Lesson_09_Activity.ipynb` in [Google Colab](https://colab.research.google.com) (**File → Upload notebook**, or open it from GitHub).
2. Run the first code cell. It installs everything you need.
3. That's it. The API key is entered inside the notebook (see below).

To run the `.py` file in Colab, upload it to the **Files** panel and run it in a code cell:

```python
!pip -q install langchain langgraph langchain-google-genai
import os, getpass
os.environ["GEMINI_API_KEY"] = getpass.getpass("Enter your Gemini API key: ")
!python lesson_09_multi_agent.py --test
```

### Option B: Your own computer

You need **Python 3.10 or newer**. Check with:

```bash
python --version
```

Create a virtual environment and install the packages:

```bash
# Go to this activity's folder
cd Week-02/Activity-04-Research-Team-Agent

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

# Install the packages
pip install langchain langgraph langchain-google-genai requests jupyter
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

> **About API usage:** one team run makes roughly 8–14 model calls (more if the Reviewer asks for revisions), plus a few embedding calls. If you hit the free-tier rate limit, wait a minute between runs.

---

## How to run the notebook

**In Colab:** open the notebook and run the cells from top to bottom with **Shift + Enter**.

**On your computer:**

```bash
jupyter notebook Lesson_09_Activity.ipynb
```

The notebook has 14 sections:

| # | Section | # | Section |
|---|---|---|---|
| 1 | Why use multiple agents? | 8 | Build the Analyst agent |
| 2 | Specialised agents and delegation | 9 | Build the Writer agent |
| 3 | The supervisor architecture | 10 | Build the Reviewer agent |
| 4 | Agent communication | 11 | Create a simple supervisor |
| 5 | Shared state | 12 | Pass information between agents using shared state |
| 6 | Design the research team | 13 | Run and test the complete workflow |
| 7 | Build the Researcher agent | 14 | Challenge |

Each agent is tested **on its own** in Sections 7–10 before it joins the team. In Section 12, `app.stream(...)` shows every step as it happens: which agent ran, and which fields of the shared state it wrote.

---

## How to complete and run the `.py` file

After the notebook, open `lesson_09_multi_agent.py`. The tools, prompts, output formats and test runner are written for you. Complete the seven `TODO` sections:

| TODO | Where | What you do |
|---|---|---|
| 1 | `ResearchState` | **Shared state:** add the seven fields the team reads and writes |
| 2 | `researcher()` | **Agent:** run the research agent and return its notes |
| 3 | `analyst()` | **Agent:** build the structured-output chain and return the analysis |
| 4 | `writer()` | **Agent:** count the draft and return it |
| 5 | `reviewer()` | **Agent:** return the review, including which draft it reviewed |
| 6 | `supervisor()` | **Delegation:** add the rules that choose the next agent |
| 7 | `build_graph()` | **Workflow:** connect the supervisor to every agent, and every agent back to the supervisor |

If you run the file before finishing a `TODO`, it stops with a message naming the `TODO` you still need to complete. When the error happens inside an agent, LangGraph adds a line like `During task with name 'analyst'` after it. The TODO message is the line just above.

Run every command from inside the activity folder.

**Step 1: Check the structure and the supervisor (TODOs 1, 6 and 7).** No API key is needed.

```bash
python lesson_09_multi_agent.py --graph               # print the graph diagram
python lesson_09_multi_agent.py --check-supervisor    # test the supervisor's 8 rules
```

Paste the diagram into [mermaid.live](https://mermaid.live) to check the hub-and-spoke shape. All 8 supervisor checks should pass.

**Step 2: Research one topic (all seven TODOs).**

```bash
python lesson_09_multi_agent.py --topic "How do AI agents use memory?"
```

You'll see the supervisor delegate each step, then the team log and the final report.

**Step 3: Run the full test.**

```bash
python lesson_09_multi_agent.py --test
```

This runs the 8 supervisor checks, then the full team on two research questions with 4 checks each:

| Check | What it tests |
|---|---|
| Every agent took part | The supervisor delegated to all four specialists |
| The Researcher used tools | Facts came from searches, not from memory |
| The final report has a Sources section | The Writer cited its sources |
| Revisions stayed within the limit | The stopping rule worked |

It saves **`research_team_results.json`** (team logs, notes, analyses, reviews and reports) and **`research_report.md`** (the first report).

The agents are LLMs, so results vary between runs. If a check shows `CHECK`, read the team log: it shows which agent did what.

---

## Challenge

The challenge is in **Section 14** of the notebook.

**Part A: Add a new specialist.** Add a **Summariser** agent that writes a three-bullet TL;DR after the Reviewer approves the report. Update the shared state, the supervisor's rules, the supervisor tests and the graph.

**Part B: An LLM supervisor.** Replace the rule-based supervisor with an LLM that chooses the next agent with structured output, but keep the revision limit as a rule in code.

**Stretch:** split the Researcher into two researchers that search in parallel and write to a shared list with the `operator.add` reducer.

---

## What to submit

Submit these through the activity's submission form on the course website:

1. **`Lesson_09_Activity.ipynb`**: run all cells, including the challenge, so the outputs are saved in the notebook. In Colab, use **File → Download → Download .ipynb**. Make sure no API key appears anywhere in it.
2. **`lesson_09_multi_agent.py`**: with all seven TODOs completed.
3. **`research_team_results.json`** and **`research_report.md`**: created by running `python lesson_09_multi_agent.py --test`.
4. **Your graph diagram**: a screenshot from [mermaid.live](https://mermaid.live) of your final graph, including the Summariser from Part A.
5. **A short reflection (100–150 words)** answering:
   - Did the Reviewer ask for changes in any of your runs? What did it ask for, and did the Writer fix it?
   - Compare the rule-based supervisor with your LLM supervisor from Part B. Which would you trust in a real product, and why?
   - Name one task where a single agent would be a better choice than a team.

### Checklist before you submit

- [ ] All seven TODOs in the `.py` file are complete
- [ ] `python lesson_09_multi_agent.py --check-supervisor` passes 8/8
- [ ] `python lesson_09_multi_agent.py --test` runs without errors
- [ ] `research_team_results.json` shows at least 14 of 16 checks passing
- [ ] Both challenge parts are completed in the notebook, with outputs visible
- [ ] Your API key does not appear in any file

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `GEMINI_API_KEY is not set` | Set the environment variable in the **same** terminal you run the file from (see API key setup). |
| `NotImplementedError: TODO ...` | You haven't finished that TODO yet. Complete it and run the file again. |
| `TODO 7a: the supervisor can't reach ...` | The conditional edge from `supervisor` must list every agent and `"finish"`. |
| `GraphRecursionError` | The team looped too many times. Check the supervisor's revision-limit rule (TODO 6) and that the Writer adds 1 to `revision` (TODO 4). |
| The Reviewer is asked to review the same draft again and again | The review must include `"draft_number": state["revision"]` (TODO 5). The supervisor uses it to know the draft was reviewed. |
| `TypeError: can only concatenate list (not "str") to list` | An agent returned `team_log` as a string. Return a **list** with one line, e.g. `["Analyst: found 4 key points."]`. |
| `Wikipedia search failed` in the notes | Check your internet connection. The Researcher can still use the course notes. |
| `503 UNAVAILABLE` or `429 RESOURCE_EXHAUSTED` | Gemini is busy or you hit the rate limit. Wait a minute and try again. |
| `404` model not found | Set `GEMINI_MODEL` to a current model name from Google AI Studio. |
