# Week 4 · Activity 02: Agent Evaluation

**Lesson 11: Agent Evaluation & Observability**

> **Building an agent is only half the job. Evaluating whether it actually works is the other half.**

In this activity you test a **Course Assistant agent** against 10 predefined scenarios and find out exactly where, and why, it fails. You'll check its answers, its tool choices, its retrieval and its honesty; trace every step it takes; turn the results into metrics; and compare the automatic results with your own judgement.

The agent under test is built with LangChain's `create_agent` (a LangGraph agent) and has two tools: a search over the course notes (Lessons 5 and 8) and a safe calculator (Week 1).

## What you'll learn

- Why agents need systematic evaluation, not just a few manual tries
- How to measure task success, tool accuracy, retrieval quality and hallucinations
- How to design test scenarios, including edge cases and hallucination traps
- How to use traces to find the cause of a failure
- How to calculate simple metrics and run a small human evaluation

## What's in this folder

| File | What it is |
|---|---|
| `Lesson_11_Activity.ipynb` | The guided notebook. It explains each idea and builds the evaluation step by step. **Start here.** |
| `lesson_11_evaluation.py` | A partially completed starter file. You complete seven `TODO` sections to build the same evaluation yourself. |
| `evaluation_dataset.json` | The 10 test scenarios, with the expected behaviour for each |
| `README.md` | This file |

Running the evaluation creates:

| File | What it is |
|---|---|
| `evaluation_results.json` | Every result, with metrics and failures grouped by reason |
| `evaluation_report.md` | A readable summary table, ready to share |
| `traces.jsonl` | One line per test case: every step the agent took, with timing and tokens |
| `human_evaluation.json` | Your own scores, created by `--human` |

## Tools used

| Tool | Purpose |
|---|---|
| Python 3.10+ | The programming language |
| `langchain` and `langgraph` | The agent under test (`create_agent`) and its tools |
| `langchain-google-genai` | Gemini (`gemini-3.8-flash`) for the agent and the LLM judge, and Gemini embeddings for the course-notes search |
| `json`, `time` (built into Python) | The dataset, results, traces and timing |

No evaluation framework is needed. Every check is a few lines of Python you can read.

---

## The evaluation dataset

`evaluation_dataset.json` holds 10 scenarios across seven categories:

| Id | Category | What it tests |
|---|---|---|
| `kb-01`, `kb-02` | knowledge | The core job: answering from the course notes |
| `tool-01` | tool_use | Using the calculator for a cost calculation |
| `tool-02` | multi_step | Finding a number in the notes, then calculating with it |
| `chat-01` | no_tool | Not using tools for a simple greeting |
| `edge-01` | edge_case | A question full of typos |
| `edge-02` | edge_case | Dividing by zero: handling a tool error honestly |
| `halluc-01` | hallucination_trap | A question the notes can't answer: the agent must not invent a price |
| `halluc-02` | hallucination_trap | A false premise the agent must correct |
| `safety-01` | safety | A prompt-injection attempt mixed with a real question |

Each case describes the expected behaviour in plain English, plus checkable rules:

```json
{
  "id": "tool-01",
  "category": "tool_use",
  "input": "An agent runs 1,250 times a day and each run costs $0.004. What does it cost for a 30-day month?",
  "expected_behavior": "Uses the calculator (1250 * 0.004 * 30) and answers $150.",
  "expected_tools": ["calculate"],
  "expected_sources": [],
  "must_include": ["150"],
  "must_not_include": [],
  "should_decline": false
}
```

In `must_include`, a `|` separates alternatives. For example, `"step budget|step limit"` passes if **either** phrase appears in the answer.

---

## How the evaluation works

For every test case, the evaluation:

1. **Runs** the agent on the input, timing it and recording a step-by-step trace.
2. **Checks** the result four ways:

| Check | Question | Fails with reason |
|---|---|---|
| Task success | Does the answer contain every `must_include` rule and no `must_not_include` phrase? | `wrong answer` |
| Tool accuracy | Is the set of tools used exactly the set expected? | `wrong tools` |
| Retrieval quality | Did the search find every expected course note? (skipped if none expected) | `missed source` |
| Hallucination | Does an LLM judge find every claim supported by what the tools returned? (only when a tool was used) | `hallucination` |

3. **Passes** the case only if there are no failure reasons. A crash is recorded with the reason `crashed`.
4. **Reports** the metrics, the failures grouped by reason, and the details of every failure.

---

## Setup

### Option A: Google Colab (easiest)

1. Open `Lesson_11_Activity.ipynb` in [Google Colab](https://colab.research.google.com) (**File → Upload notebook**, or open it from GitHub).
2. Run the first code cell. It installs everything you need.
3. The notebook writes its own copy of `evaluation_dataset.json` if it isn't there, so you can open the notebook on its own.

To run the `.py` file in Colab, upload **both** `lesson_11_evaluation.py` and `evaluation_dataset.json` to the **Files** panel, then run in a code cell:

```python
!pip -q install langchain langgraph langchain-google-genai
import os, getpass
os.environ["GEMINI_API_KEY"] = getpass.getpass("Enter your Gemini API key: ")
!python lesson_11_evaluation.py --run
```

The `--human` mode asks you questions, so it needs a terminal. In Colab, use Section 14 of the notebook instead.

### Option B: Your own computer

You need **Python 3.10 or newer**. Check with:

```bash
python --version
```

Create a virtual environment and install the packages:

```bash
# Go to this activity's folder
cd Week-04/Activity-02-Agent-Evaluation

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

**Optional:** to use a different Gemini model, set `GEMINI_MODEL` the same way. The default is `gemini-3.8-flash`.

> **About API usage:** a full run makes roughly 25–35 model calls: the agent's steps plus one judge call per case that used a tool. Use `--no-judge` to skip the judge, or `--only CASE_ID` to run a single case. If you hit the free-tier rate limit, wait a minute and run again.

---

## How to run the notebook

**In Colab:** open the notebook and run the cells from top to bottom with **Shift + Enter**.

**On your computer:**

```bash
jupyter notebook Lesson_11_Activity.ipynb
```

The notebook has 15 sections:

| # | Section | # | Section |
|---|---|---|---|
| 1 | Why agent evaluation matters | 9 | Compare expected vs actual outputs |
| 2 | Task success and failure | 10 | Evaluate tool selection and task success |
| 3 | Tool accuracy | 11 | Identify failure cases |
| 4 | Retrieval quality | 12 | Add basic tracing and logging |
| 5 | Hallucinations | 13 | Calculate simple evaluation metrics |
| 6 | Designing evaluation scenarios | 14 | Perform a small human evaluation |
| 7 | Create an evaluation dataset | 15 | Challenge |
| 8 | Run the agent against predefined test cases | | |

---

## How to complete and run the `.py` file

After the notebook, open `lesson_11_evaluation.py`. The agent under test, tracing, the LLM judge and reporting are written for you. Complete the seven `TODO` sections:

| TODO | Where | Topic | What you do |
|---|---|---|---|
| 1 | `load_dataset()` | Loading test cases | Load the JSON file and check every case has every required field |
| 2 | `run_case()` | Recording results | Record the tools used and the sources retrieved |
| 3 | `check_tools()` | Running evaluations | Compare the set of expected tools with the set used |
| 4 | `check_answer()` | Running evaluations | Check the `must_not_include` phrases |
| 5 | `evaluate_case()` | Identifying failures | Add the failure reasons |
| 6 | `calculate_metrics()` | Calculating metrics | Calculate task success, tool accuracy, retrieval hit rate and hallucination rate |
| 7 | `group_failures()` | Identifying failures | Group the failed case ids by reason |

If you run the file before finishing a `TODO`, it stops with a message naming the `TODO` you still need to complete.

Run every command from inside the activity folder.

**Step 1: Validate the dataset (TODO 1).** No API key is needed.

```bash
python lesson_11_evaluation.py --check-dataset
```

**Step 2: Try a single case and read its trace.**

```bash
python lesson_11_evaluation.py --run --only tool-02
```

This prints every step the agent took: each tool call, each tool result and the final answer.

**Step 3: Run the full evaluation (all seven TODOs).**

```bash
python lesson_11_evaluation.py --run              # with the LLM judge
python lesson_11_evaluation.py --run --no-judge   # without it (fewer API calls)
```

This saves `evaluation_results.json`, `evaluation_report.md` and `traces.jsonl`.

**Step 4: Score some answers yourself.**

```bash
python lesson_11_evaluation.py --human
```

You score four answers from 1 to 5 (up to two that failed and some that passed). It reports how often you agreed with the automatic result and saves `human_evaluation.json`.

---

## How to interpret the results

The report looks like this (your numbers will differ):

```text
Case       Category            Task  Tools  Source  Judge  Result
kb-01      knowledge           ok    ok     ok      ok     PASS
edge-02    edge_case           FAIL  ok     -       ok     FAIL: wrong answer
halluc-01  hallucination_trap  FAIL  ok     -       HALL   FAIL: wrong answer, hallucination
...
Pass rate:            80.0%  (8/10 cases)
Task success rate:    80.0%
Tool accuracy:        100.0%
Retrieval hit rate:   100.0%
Hallucination rate:   11.1%  (lower is better)
```

A `-` means the check didn't apply to that case (for example, no sources were expected).

| Metric | Meaning | Better is |
|---|---|---|
| Pass rate | Share of cases that passed **every** check | Higher |
| Task success rate | Share of answers that met their must-include and must-not-include rules | Higher |
| Tool accuracy | Share of cases that used exactly the expected tools | Higher |
| Retrieval hit rate | Of the cases that needed specific notes, the share that found them | Higher |
| Hallucination rate | Of the judged answers, the share with unsupported claims | **Lower** |
| Average latency | Seconds per case | Lower |

**Use the failure reason to decide what to fix:**

| Reason | Likely cause | Where to look in the trace | Typical fix |
|---|---|---|---|
| `wrong tools` | Tool selection | The TOOL CALL steps | Clearer tool descriptions or system prompt |
| `missed source` | Retrieval | The search TOOL RESULT | Better queries, more results (`k`), better documents |
| `wrong answer` with the right tools and sources | Generation | The final ANSWER | A clearer system prompt |
| `wrong answer` that looks correct to you | The **test** | Compare with `expected_behavior` | Add a phrase to `must_include` |
| `hallucination` | Answering beyond the sources | The final ANSWER vs the TOOL RESULTs | A stricter "answer only from the notes" rule |

**Keep in mind:**

- **The sample is tiny.** With 10 cases, one case is 10 percentage points.
- **Check categories, not just the total.** Failing every hallucination trap is serious even with a high overall score.
- **LLMs vary.** Run the evaluation more than once before drawing conclusions. A case that sometimes fails is "flaky".
- **Automatic checks can be wrong too.** Keyword rules and LLM judges make mistakes. That's why the human evaluation exists.
- **Re-run after every change.** Evaluation is most useful for catching **regressions**: things that used to work and broke.

---

## Challenge

The challenge is in **Section 15** of the notebook.

**Part A: Write new test cases.** Add a multi-step case and a hallucination trap of your own design to `evaluation_dataset.json`, and try to write a trap the agent fails.

**Part B: Fix a failure, then check for regressions.** Use a trace to find the cause of one failure, improve the agent's prompt or a tool description, re-run the **whole** evaluation, and compare the metrics before and after.

**Stretch:** run the evaluation three times and find the flaky cases.

---

## What to submit

Submit these through the activity's submission form on the course website:

1. **`Lesson_11_Activity.ipynb`**: run all cells, including your human scores and the challenge, so the outputs are saved. In Colab, use **File → Download → Download .ipynb**. Make sure no API key appears anywhere in it.
2. **`lesson_11_evaluation.py`**: with all seven TODOs completed.
3. **`evaluation_dataset.json`**: with your two new test cases from Part A.
4. **`evaluation_results.json`**, **`evaluation_report.md`** and **`traces.jsonl`**: from `python lesson_11_evaluation.py --run`.
5. **`human_evaluation.json`**: from `python lesson_11_evaluation.py --human`.
6. **A short reflection (150–200 words)** answering:
   - Which case failed, what did its trace show, and was it a tool, retrieval or generation problem?
   - Did you disagree with any automatic result? Who was right, and how would you improve the test?
   - In Part B, did your fix improve the metrics? Did anything get worse?

### Checklist before you submit

- [ ] All seven TODOs in the `.py` file are complete
- [ ] `--check-dataset` loads 12 cases (the original 10 plus your 2)
- [ ] `--run` finishes and saves the three output files
- [ ] You've completed the human evaluation (notebook Section 14 and `--human`)
- [ ] Both challenge parts are completed in the notebook, with outputs visible
- [ ] Your API key does not appear in any file

You don't need a perfect pass rate. A well-explained failure is worth more than an unexplained pass.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `GEMINI_API_KEY is not set` | Set the environment variable in the **same** terminal you run the file from (see API key setup). |
| `NotImplementedError: TODO ...` | You haven't finished that TODO yet. Complete it and run the file again. |
| `FileNotFoundError: evaluation_dataset.json` | Run the command from the activity folder, where the dataset file is. |
| `ValueError: Test case ... is missing` | One of your new test cases is missing a field. Copy an existing case as a template. |
| `json.decoder.JSONDecodeError` | The dataset has a syntax error, often a missing comma or quote. Check the line number in the error. |
| Every case fails with `wrong tools` | Check TODO 2 (recording `tools_used`) and TODO 3 (comparing **sets**). |
| `retrieval_hit_rate` shows `n/a` | No case expected sources, or TODO 2 isn't recording `sources_retrieved`. |
| `--human` says to run the evaluation first | Run `python lesson_11_evaluation.py --run` (without `--only`) first. |
| Results change between runs | That's normal for LLMs, and it's exactly why we evaluate. Run it a few times. |
| `503 UNAVAILABLE` or `429 RESOURCE_EXHAUSTED` | Gemini is busy or you hit the rate limit. Wait a minute, use `--no-judge`, or run fewer cases with `--only`. |
| `404` model not found | Set `GEMINI_MODEL` to a current model name from Google AI Studio. |
