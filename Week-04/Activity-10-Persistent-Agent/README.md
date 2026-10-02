# Week 4 · Lab 10: Persistent Agent

**Lesson 10: Agent State, Memory & Persistence**

Build a **Learning Plan agent** that writes a personalised study plan one module at a time, and **saves its progress after every step**. You'll interrupt it halfway through on purpose, restart the program, and watch it recover and continue exactly where it stopped, without repeating or losing any work.

The agent keeps two kinds of state in two local files you can open and inspect:

| | File | Example |
|---|---|---|
| **User state** | `user_profiles.json` | Amara is a beginner who studies 30 minutes a day and has completed 1 plan |
| **Conversation state** | `checkpoints.db` | Thread `amara-1`: 2 of 4 modules written; next step: write module 3 |

This activity builds on the **Lesson 8 LangGraph activity**, where you used `InMemorySaver`. Here you switch to `SqliteSaver`, so checkpoints survive the program stopping.

## What you'll learn

- What agent state is, and the difference between user state and conversation state
- The agent lifecycle: created, running, interrupted, resumed, completed
- Why persistence matters, and what a checkpoint stores
- How to add SQLite checkpointing to a LangGraph agent
- How to recover after an interruption with `app.invoke(None, config)`
- How to keep multiple users and conversations separate

## What's in this folder

| File | What it is |
|---|---|
| `Lesson_10_Activity.ipynb` | The guided notebook. It explains each idea and builds the full agent step by step. **Start here.** |
| `lesson_10_persistent_agent.py` | A partially completed starter file. You complete seven `TODO` sections to build the same agent yourself. |
| `README.md` | This file |

Running the code creates `checkpoints.db` and `user_profiles.json`. The test uses its own files (`test_checkpoints.db`, `test_user_profiles.json`) and creates `persistence_test_results.json`, which you submit.

## Tools used

| Tool | Purpose |
|---|---|
| Python 3.10+ | The programming language |
| `langgraph` | State, nodes, edges and checkpointing |
| `langgraph-checkpoint-sqlite` | `SqliteSaver`: saves checkpoints to a local SQLite file |
| `langchain` and `langchain-google-genai` | Prompts, structured output and Gemini (`gemini-3.8-flash`) |
| `sqlite3` and `json` (built into Python) | Local storage, with no server or database to install |

---

## How the workflow works

```text
                 START
                   │
          route_start: is this thread's plan already done?
             │ no                         │ yes
             ▼                            ▼
       load_profile                     chat ──► END     (follow-up questions)
             ▼
       plan_outline
             ▼
       write_module ◄─┐   one module per step,
             │        │   so every module gets its own checkpoint
             ├────────┘   (loops until all modules are written)
             ▼
         finalize ──► END
```

| Step | What it does | State it writes |
|---|---|---|
| `load_profile` | Copies the user's profile (user state) into the thread | `profile`, `status` |
| `plan_outline` | Plans the module titles with structured output | `modules`, `current_module`, `status` |
| `write_module` | Writes **one** module, then loops | `lessons` (adds one), `current_module` |
| `finalize` | Assembles the plan and updates the user's profile | `final_plan`, `status`, `messages` |
| `chat` | Answers follow-up questions about a finished plan | `messages` |

**What happens when the agent is interrupted:**

1. LangGraph saves a **checkpoint** after every step, in `checkpoints.db`, under the task's **thread id** (for example `amara-1`).
2. If the program stops while writing module 3, modules 1 and 2 are already saved, and the checkpoint records that the **next step** is `write_module`.
3. After a restart, `app.get_state(config)` shows the saved progress, and `app.invoke(None, config)` continues from the next step. Finished steps are never repeated.

The interruption is simulated: setting `CRASH_AT_MODULE` (or `--crash-at`) makes `write_module` raise an error just before writing that module.

---

## Setup

### Option A: Google Colab (easiest)

1. Open `Lesson_10_Activity.ipynb` in [Google Colab](https://colab.research.google.com) (**File → Upload notebook**, or open it from GitHub).
2. Run the first code cell. It installs everything you need.
3. That's it. The API key is entered inside the notebook (see below).

`checkpoints.db` and `user_profiles.json` appear in Colab's **Files** panel. They last while the runtime is running. Download them if you want to keep them.

To run the `.py` file in Colab, upload it to the **Files** panel and run it in code cells. Each `!python` command runs as a **separate program**, which is a real restart:

```python
!pip -q install langchain langgraph langchain-google-genai langgraph-checkpoint-sqlite
import os, getpass
os.environ["GEMINI_API_KEY"] = getpass.getpass("Enter your Gemini API key: ")
!python lesson_10_persistent_agent.py --test
```

### Option B: Your own computer

You need **Python 3.10 or newer**. Check with:

```bash
python --version
```

Create a virtual environment and install the packages:

```bash
# Go to this activity's folder
cd Week-04/Activity-01-Persistent-Agent

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

# Install the packages
pip install langchain langgraph langchain-google-genai langgraph-checkpoint-sqlite jupyter
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

**In Colab:** open the notebook and run the cells from top to bottom with **Shift + Enter**. Running the notebook from the top deletes the old `checkpoints.db` and `user_profiles.json`, so you always start fresh.

**On your computer:**

```bash
jupyter notebook Lesson_10_Activity.ipynb
```

The notebook has 13 sections:

| # | Section | # | Section |
|---|---|---|---|
| 1 | What is agent state? | 8 | Add checkpointing with LangGraph |
| 2 | User state vs conversation state | 9 | Run the agent and save its state |
| 3 | The agent lifecycle | 10 | Simulate an interruption |
| 4 | Why persistence matters | 11 | Recover the previous state and continue the task |
| 5 | Checkpoints | 12 | Test multiple conversations and users |
| 6 | Saving and recovering state | 13 | Challenge |
| 7 | Build a simple agent with structured state | | |

In Section 11, the notebook closes the database connection and builds a brand-new app before resuming. That proves the state comes from the file, not from Python's memory.

---

## How to complete and run the `.py` file

After the notebook, open `lesson_10_persistent_agent.py`. The prompts, nodes, helpers and test are written for you. Complete the seven `TODO` sections:

| TODO | Where | Topic | What you do |
|---|---|---|---|
| 1 | `PlanState` | State definition | Add the seven fields, including the `operator.add` reducer for `lessons` |
| 2 | `get_checkpointer()` | Checkpointing | Open a SQLite connection and wrap it in a `SqliteSaver` |
| 3 | `save_profile()` | Persistence | Save user state to `user_profiles.json` |
| 4 | `route_modules()` | Agent execution | Loop back to `write_module` until every module is written |
| 5 | `build_graph()` | Checkpointing | Add the edges and compile **with** the checkpointer |
| 6 | `start_task()` | Agent execution | Run the graph on a thread |
| 7 | `resume_task()` | Recovery | Load the saved state, then continue with `app.invoke(None, config)` |

If you run the file before finishing a `TODO`, it stops with a message naming the `TODO` you still need to complete. When the error happens inside a node, LangGraph adds a line like `During task with name 'write_module'` after it. The TODO message is the line just above.

Run every command from inside the activity folder.

**Step 1: Check the structure (TODOs 1, 2 and 5).** No API key is needed.

```bash
python lesson_10_persistent_agent.py --graph
```

Paste the diagram into [mermaid.live](https://mermaid.live). Check the loop on `write_module`.

**Step 2: Start a task and interrupt it.** Each command below is a separate run of the program, so this is a **real** restart.

```bash
python lesson_10_persistent_agent.py --start amara "Learn SQL for data analysis" --crash-at 3
```

The program stops while writing module 3 and tells you the thread id (`amara-1`).

**Step 3: Look at what was saved.** No API key is needed.

```bash
python lesson_10_persistent_agent.py --threads           # list every saved thread
python lesson_10_persistent_agent.py --status amara-1    # 2 of 4 modules done, next step: write_module
```

**Step 4: Recover and continue.**

```bash
python lesson_10_persistent_agent.py --resume amara-1
```

The agent continues at module 3 and finishes the plan. Open `user_profiles.json`: Amara now has 1 completed plan.

**Step 5: Come back later with a question.**

```bash
python lesson_10_persistent_agent.py --ask amara-1 "Which module should I start with?"
```

**Step 6: Run the full test.**

```bash
python lesson_10_persistent_agent.py --test
```

The test uses its own files, so it doesn't touch your `checkpoints.db`. It runs 12 checks:

| Part | What it tests |
|---|---|
| Interrupt | Amara's task stops at module 3; 2 modules and the next step are saved |
| Restart and resume | A brand-new app, built from the file, finishes the plan with exactly 4 modules |
| Second user | Ben's plan uses Ben's profile and has its own thread |
| Follow-up | A question is added to Amara's saved conversation |
| User state | Both profiles record their completed plans; both threads are in the SQLite file |

It saves **`persistence_test_results.json`**.

---

## Challenge

The challenge is in **Section 13** of the notebook.

**Part A: An approval that survives a restart.** Add a `review_outline` node that pauses with `interrupt()` for the user to approve the plan. Restart the app while it's paused, then resume with `Command(resume="yes")`.

**Part B: Forget a user.** Write `forget_user()` that deletes all of a user's threads with `checkpointer.delete_thread()` and removes their profile from the JSON file.

**Stretch: time travel.** Resume a thread from an **earlier** checkpoint using `get_state_history()`, so the agent rewrites the last modules as a new branch.

---

## What to submit

Submit these through the activity's submission form on the course website:

1. **`Lesson_10_Activity.ipynb`**: run all cells, including the challenge, so the outputs are saved in the notebook. In Colab, use **File → Download → Download .ipynb**. Make sure no API key appears anywhere in it.
2. **`lesson_10_persistent_agent.py`**: with all seven TODOs completed.
3. **`persistence_test_results.json`**: created by running `python lesson_10_persistent_agent.py --test`.
4. **Terminal output from Steps 2–4**: a screenshot or a text file showing the interruption, `--status` and `--resume`.
5. **A short reflection (100–150 words)** answering:
   - In your own words, what is the difference between user state and conversation state in this agent?
   - After the interruption, which steps ran again and which didn't? Why?
   - Name one real app where losing an agent's state halfway through would be a serious problem.

### Checklist before you submit

- [ ] All seven TODOs in the `.py` file are complete
- [ ] `--start ... --crash-at 3` stops at module 3, and `--resume` finishes the plan without repeating modules
- [ ] `python lesson_10_persistent_agent.py --test` runs without errors
- [ ] `persistence_test_results.json` shows at least 10 of 12 checks passing
- [ ] Both challenge parts are completed in the notebook, with outputs visible
- [ ] Your API key does not appear in any file

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `GEMINI_API_KEY is not set` | Set the environment variable in the **same** terminal you run the file from (see API key setup). |
| `NotImplementedError: TODO ...` | You haven't finished that TODO yet. Complete it and run the file again. |
| `ModuleNotFoundError: langgraph.checkpoint.sqlite` | Install the extra package: `pip install langgraph-checkpoint-sqlite`. |
| `Checkpointer requires one or more of the following 'configurable' keys` | You called `.invoke()` without a thread id. Use `thread_config(thread_id)` as the second argument. |
| `--resume` says "No saved state" | Check the thread id with `--threads`, and run the command from the **same folder** as `checkpoints.db`. |
| `--resume` says "nothing left to do" | That thread already finished. Start a new task with `--start`. |
| The resumed plan has duplicate modules | `lessons` must use the `operator.add` reducer (TODO 1), and resuming must pass `None` as the input (TODO 7b), not the original request. |
| `sqlite3.ProgrammingError` about threads | Add `check_same_thread=False` to `sqlite3.connect(...)` (TODO 2). |
| `database is locked` | Another program (or a second notebook) has `checkpoints.db` open. Close it and try again. |
| `503 UNAVAILABLE` or `429 RESOURCE_EXHAUSTED` | Gemini is busy or you hit the rate limit. Wait a minute, then use `--resume`: your progress is saved. |
| `404` model not found | Set `GEMINI_MODEL` to a current model name from Google AI Studio. |
