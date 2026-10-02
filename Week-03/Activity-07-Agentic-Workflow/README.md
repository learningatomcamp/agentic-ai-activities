# Week 3 · Activity 01: Multi-Step Support Workflow

**Lesson 7: Agentic Workflows**

**Week 3 theme:** from one agent to agentic systems.

Build a support-desk workflow for an online learning platform. A customer message goes in; a checked, approved reply comes out. On the way, it is classified by an agent, routed to the right branch, enriched with data looked up in parallel, drafted, reviewed in a loop, and, for refunds, approved by a human.

## What you'll learn

You'll use all seven workflow patterns from Lesson 7:

| Pattern | Where you'll use it |
|---|---|
| **Sequential** | Triage → route → branch → review → approval, each step using the last one's output |
| **Conditional** | Escalated messages skip the rest; approval only happens when it's needed |
| **Routing** | Billing and technical messages go to different branches |
| **Loop** | A reviewer agent sends drafts back for revision, up to a limit |
| **Parallel** | Customer and order lookups run at the same time |
| **State** | One dictionary carries everything through the workflow, with a trace of every step |
| **Human-in-the-loop** | A person approves, edits or rejects every refund before it's sent |

## Architecture

```text
               Customer message
                      ↓
               Triage agent                 (LLM, structured output)
                      ↓
                   Route
              ↙               ↘
     A: Billing               B: Technical
     parallel lookups         help-doc search
     + refund policy          (no match → escalate)
              ↘               ↙
                 Draft reply                (LLM)
                      ↓
           Review agent ⟲ revise            (loop, max 2 revisions)
                      ↓
           Human approval (refunds only)
                      ↓
                    Result
```

## What's in this folder

| File | What it is |
|---|---|
| `Lesson_07_Activity.ipynb` | The guided notebook. It explains each pattern and builds the full workflow step by step. **Start here.** |
| `lesson_07_workflow.py` | A partially completed starter file. You complete five `TODO` sections to build the same workflow yourself. |
| `README.md` | This file |

Running the tests also creates `workflow_test_results.json`, which you submit.

## Tools used

| Tool | Purpose |
|---|---|
| Python 3.10+ | The programming language |
| Gemini API (`google-genai`) | The triage, writer and reviewer agents |
| `pydantic` | Structured outputs for triage and review (from Week 1) |
| `concurrent.futures` (built into Python) | Parallel execution |

No workflow frameworks are needed. Every pattern is plain Python, so you can see exactly how it works.

---

## Setup

### Option A: Google Colab (easiest)

1. Open `Lesson_07_Activity.ipynb` in [Google Colab](https://colab.research.google.com) (**File → Upload notebook**, or open it from GitHub).
2. Run the first code cell. It installs everything you need.
3. That's it. The API key is entered inside the notebook (see below).

To run the `.py` file in Colab, upload it to the Files panel (left sidebar) and run it in a code cell:

```python
!pip -q install google-genai pydantic
import os, getpass
os.environ["GEMINI_API_KEY"] = getpass.getpass("Enter your Gemini API key: ")
!python lesson_07_workflow.py --test
```

The `--run` mode asks you to approve refunds by typing, which needs a terminal. In Colab, use the approval cells in the notebook instead.

### Option B: Your own computer

You need **Python 3.10 or newer**. Check with:

```bash
python --version
```

Create a virtual environment and install the packages:

```bash
# Go to this activity's folder
cd Week-03/Activity-01-Multi-Step-Workflow

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

# Install the packages
pip install google-genai pydantic jupyter
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
jupyter notebook Lesson_07_Activity.ipynb
```

Two cells ask you to make a decision as the human approver. Type `a` to approve, `e` to edit, or `r` to reject.

The notebook has 15 sections:

| # | Section | # | Section |
|---|---|---|---|
| 1 | From one agent to agentic systems | 9 | Branch A: billing |
| 2 | The building blocks | 10 | Branch B: technical |
| 3 | Setup | 11 | Loops: review and revise |
| 4 | The project: a support desk | 12 | Human-in-the-loop |
| 5 | State | 13 | Putting it all together |
| 6 | Sequential workflows: the triage agent | 14 | Test the workflow |
| 7 | Conditional workflows and routing | 15 | Challenge |
| 8 | Parallel execution | | |

---

## How to complete and run the `.py` file

After the notebook, open `lesson_07_workflow.py`. The mock data, LLM helpers, triage agent, branches and reviewer are written for you. Complete the five `TODO` sections that connect them:

| TODO | Where | Pattern | What you do |
|---|---|---|---|
| 1 | `route()` | Routing | Choose the branch from the triage result and save it in the state |
| 2 | `gather_billing_context()` | Parallel | Run the customer and order lookups at the same time with `ThreadPoolExecutor` |
| 3 | `review_loop()` | Loop | Stop when approved or at the revision limit; otherwise revise and repeat |
| 4 | `human_approval()` | Human-in-the-loop | Act on the approve, edit or reject decision |
| 5 | `run_workflow()` | Sequential + conditional | Connect every step in the right order |

If you run the file before finishing a `TODO`, it stops with a message naming the `TODO` you still need to complete.

Run every command from inside the activity folder.

**Step 1: Test parallel execution (TODO 2).** No API key is needed.

```bash
python lesson_07_workflow.py --parallel-demo
```

You should see about **2 seconds** one after the other and about **1 second** in parallel.

**Step 2: Run one message (all five TODOs).** You act as the human approver for refunds.

```bash
python lesson_07_workflow.py --run "Can I get a refund for order A-1001?" --customer C001
python lesson_07_workflow.py --run "My videos keep buffering" --customer C003
```

Each run prints a trace: every step, how long it took, and what happened.

| Customer | Order | Refund policy |
|---|---|---|
| `C001` Lena | `A-1001`, bought 5 days ago, 10% done | Eligible |
| `C002` Kenji | `A-1002`, bought 40 days ago, 85% done | Not eligible |
| `C003` Maria | none | none |

**Step 3: Run all the tests.**

```bash
python lesson_07_workflow.py --test
```

This runs four test cases, one for each path through the workflow, approving refunds automatically. It checks 14 things and saves the results to **`workflow_test_results.json`**.

| Test | Path |
|---|---|
| Eligible refund | Billing → parallel lookups → review → **human approval** → sent |
| Refund outside policy | Billing → policy says no → review → sent (no approval needed) |
| Help article found | Technical → help article → review → sent |
| No help article | Technical → **early exit** → escalated |

The LLM steps can behave a little differently on each run. If a check shows `CHECK`, read the trace for that test to find the step that went a different way.

---

## Challenge

The challenge is in **Section 15** of the notebook.

**Part A: Add a third route.** Add a `feedback` category and branch that saves feedback to `feedback.json` and sends a thank-you reply. Test it with *"I love the new dark mode! Could you add subtitles in Spanish too?"*

**Part B: An urgency rule.** Make every message with `urgency: "high"` go to a human for approval, whichever branch it took.

**Stretch: Parallel reviewers.** Replace the single reviewer with two reviewers running in parallel, one for tone and one for accuracy. A draft is approved only if both approve.

---

## What to submit

Submit these through the activity's submission form on the course website:

1. **`Lesson_07_Activity.ipynb`**: run all cells, including the challenge, so the outputs are saved in the notebook. In Colab, use **File → Download → Download .ipynb**. Make sure no API key appears anywhere in it.
2. **`lesson_07_workflow.py`**: with all five TODOs completed.
3. **`workflow_test_results.json`**: created by running `python lesson_07_workflow.py --test`.
4. **A short reflection (150–200 words)** answering:
   - Pick one test case and walk through its trace. Which patterns did that message pass through?
   - Why is the refund policy written in Python instead of being left to the LLM?
   - Where else in a real product would you add a human-in-the-loop step, and why?

### Checklist before you submit

- [ ] All five TODOs in the `.py` file are complete
- [ ] `python lesson_07_workflow.py --parallel-demo` shows parallel is faster
- [ ] `python lesson_07_workflow.py --test` runs without errors
- [ ] `workflow_test_results.json` shows at least 12 of 14 checks passing
- [ ] Challenge Parts A and B are completed in the notebook, with outputs visible
- [ ] Your API key does not appear in any file

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `GEMINI_API_KEY is not set` | Set the environment variable in the **same** terminal you run the file from (see API key setup). |
| `NotImplementedError: TODO ...` | You haven't finished that TODO yet. Complete it and run the file again. |
| `--parallel-demo` shows the same time for both | In TODO 2, call `submit()` for **both** lookups before calling `.result()` on either. Calling `.result()` straight after the first `submit()` makes it wait. |
| The workflow never ends | Check TODO 3: the loop must `break` when approved **and** when `revisions` reaches `MAX_REVISIONS`. |
| The refund test never asks for approval | Check TODO 5: run `human_approval()` when `state["needs_approval"]` is `True`. |
| The escalate test runs the review loop | Check TODO 5: skip review and approval when `state["status"] == "escalated"`. |
| `503 UNAVAILABLE` or `429 RESOURCE_EXHAUSTED` | Gemini is busy or you hit the rate limit. The code retries automatically; if it keeps failing, wait a minute. |
| `404` model not found | Set `GEMINI_MODEL` to a current model name from Google AI Studio. |
