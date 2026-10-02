# Week 4 · Lab 12: Break the Agent

**Lesson 12: Agent Security & Responsible AI**

> **Break it safely → understand why it failed → fix it.**

In this activity you attack a deliberately **vulnerable** Helpdesk agent with a set of safe, prepared tests, watch the attacks succeed, record what happened, then add security controls and re-test until the attacks stop working.

You'll see the main agent vulnerabilities first-hand: prompt injection, tool misuse, data leakage and excessive agency. Then you'll fix each one with a control in **code**, because a prompt alone doesn't hold up under attack.

## ⚠️ Safety first

Everything in this activity runs **locally, on fake, in-memory data**:

- No real customers, emails, money or systems. The "email" and "refund" tools only append to a Python list.
- The data **resets before every attack**, so nothing you do persists.
- There are **no real credentials** anywhere, and **no destructive actions** against anything real.

**The one rule that matters:** only ever run attacks like these against systems **you own and are authorised to test**. Using these techniques on anyone else's agent or system is harmful and usually illegal. This lab exists so you can learn to defend agents, by safely attacking one you own.

## What you'll learn

- Why agent security is harder than ordinary software security
- The main vulnerabilities: prompt injection (direct and indirect), tool misuse, data leakage, excessive agency
- Permission boundaries, authentication vs authorization, sandboxing and human approval
- How to test an agent against attacks, record findings, fix them, and prove the fix with a re-test

## What's in this folder

| File | What it is |
|---|---|
| `Lesson_12_Activity.ipynb` | The guided notebook. It explains each vulnerability, runs the attacks, and shows the fixes. **Start here.** |
| `lesson_12_vulnerable_agent.py` | The vulnerable agent. You complete five `TODO` security controls to fix it. |
| `security_test_cases.json` | 10 safe attack scenarios, each with the expected secure behaviour |
| `README.md` | This file |

Running the comparison creates `security_results.json`, which you submit.

## Tools used

| Tool | Purpose |
|---|---|
| Python 3.10+ | The programming language |
| `langchain` and `langgraph` | The agent, its tools, the permission gate and the human-approval gate |
| `langchain-google-genai` | Gemini (`gemini-3.8-flash`) |
| `json`, `re` (built into Python) | The test cases, the sandbox data, and input validation |

---

## The agent and its vulnerabilities

The **Helpdesk agent** has eight tools: answer support hours, look up one customer, read the whole customer database, read a ticket, check an order, send an email, issue a refund, and delete all tickets. The current user is a low-privilege `support_agent`.

It has **five security controls**, all switched **OFF** to start with. Each attack in the dataset targets a vulnerability that one or more controls will fix:

| Control | Fixes |
|---|---|
| `harden_system_prompt` | Direct prompt injection; system-prompt leakage |
| `treat_tool_output_as_data` | Indirect injection (hidden instructions in a ticket or order) |
| `restrict_data_access` | Bulk data dumps; unvalidated lookups |
| `require_permissions` | Destructive/manager-only tools; privilege escalation |
| `require_human_approval` | High-impact actions (refunds) running with no human |

The controls live **inside the tools and the workflow**, not in the prompt, because the model is exactly the part an attacker manipulates.

---

## How the before/after test works

```text
run every attack with controls OFF   ->   count how many the attacker wins   (vulnerable)
run every attack with controls ON    ->   count again                        (secured)
compare: which attacks did the controls fix?
```

The test judges each attack mainly by **what the tools actually did** (emails sent, refunds issued, tickets deleted, customer records exposed), not by what the agent *said*. An agent can claim it refused while a tool already ran, so actions are the real evidence.

---

## Setup

### Option A: Google Colab (easiest)

1. Open `Lesson_12_Activity.ipynb` in [Google Colab](https://colab.research.google.com) (**File → Upload notebook**, or open it from GitHub).
2. Run the first code cell. It installs everything you need.
3. The notebook writes its own copy of `security_test_cases.json` if it isn't there.

To run the `.py` file in Colab, upload **both** `lesson_12_vulnerable_agent.py` and `security_test_cases.json` to the **Files** panel, then run in a code cell:

```python
!pip -q install langchain langgraph langchain-google-genai
import os, getpass
os.environ["GEMINI_API_KEY"] = getpass.getpass("Enter your Gemini API key: ")
!python lesson_12_vulnerable_agent.py --compare
```

### Option B: Your own computer

You need **Python 3.10 or newer**. Check with:

```bash
python --version
```

Create a virtual environment and install the packages:

```bash
cd Week-04/Activity-03-Break-the-Agent
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
pip install langchain langgraph langchain-google-genai jupyter
```

---

## API key setup

Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com): sign in and choose **Get API key**.

**Never paste your key into a code cell or a `.py` file**, and never commit it to GitHub.

### For the notebook

Run the API key cell and paste your key into the box. It's hidden and isn't saved in the notebook.

### For the `.py` file

```bash
# macOS / Linux
export GEMINI_API_KEY="paste-your-key-here"
# Windows (Command Prompt)
set GEMINI_API_KEY=paste-your-key-here
# Windows (PowerShell)
$env:GEMINI_API_KEY="paste-your-key-here"
```

Set it again in each new terminal. To use another model, set `GEMINI_MODEL`. The default is `gemini-3.8-flash`.

> **API usage:** one full `--compare` runs 20 attacks (10 before, 10 after), so roughly 25–45 model calls. If you hit the free-tier rate limit, wait a minute, or run single attacks with `--attack`.

---

## How to run the notebook

**In Colab:** run the cells from top to bottom with **Shift + Enter**.

**On your computer:**

```bash
jupyter notebook Lesson_12_Activity.ipynb
```

The notebook has 16 sections:

| # | Section | # | Section |
|---|---|---|---|
| 1 | Why agent security matters | 9 | Human approval |
| 2 | Prompt injection | 10 | Introduce the vulnerable agent |
| 3 | Tool misuse | 11 | The attacks |
| 4 | Data leakage | 12 | Identify which vulnerabilities you can trigger |
| 5 | Excessive agency | 13 | Record the vulnerability, impact and behaviour |
| 6 | Permission boundaries | 14 | Modify the agent to mitigate |
| 7 | Authentication vs authorization | 15 | Re-run the tests and compare |
| 8 | Sandboxing | 16 | Challenge |

In the notebook you turn the controls on by **flipping them to `True`** to see the effect. In the `.py` file, you **write the controls yourself**.

---

## How to complete and run the `.py` file

Open `lesson_12_vulnerable_agent.py`. The agent, the attacks and the scoring are written for you. Complete the five `TODO` security controls. Each one is already wired into the agent behind an `if CONTROLS[...]` switch; your job is to write what happens when the switch is on.

| TODO | Control | Where | What you implement |
|---|---|---|---|
| 1 | Harden the system prompt | `system_prompt()` | Return the hardened prompt when the control is on |
| 2 | Treat tool output as data | `read_ticket()`, `check_order_status()` | Wrap untrusted record text so hidden instructions aren't obeyed |
| 3 | Restrict data access | `read_customer()`, `read_customer_database()` | Validate the id; never dump the whole table |
| 4 | Require permissions | `guarded_tools_node()` | Block manager-only tools unless the user's real role is manager |
| 5 | Require human approval | `approval_gate()` | Pause with `interrupt()`; if not approved, the action doesn't run |

The controls all start `False`, so the agent is fully vulnerable until you implement them.

Run every command from inside the activity folder.

**Step 1: See the attacks (no API key needed).**

```bash
python lesson_12_vulnerable_agent.py --list
```

**Step 2: Attack the vulnerable agent.**

```bash
python lesson_12_vulnerable_agent.py --test            # all controls OFF: most attacks win
python lesson_12_vulnerable_agent.py --attack sec-03   # watch one attack in detail
```

**Step 3: Implement the five TODOs.** As you finish each one, test it:

```bash
python lesson_12_vulnerable_agent.py --attack sec-07 --secure   # after TODO 5, this should be defended
```

**Step 4: Compare before and after.**

```bash
python lesson_12_vulnerable_agent.py --compare
```

This runs every attack with controls off, then on, prints both tables, and saves `security_results.json`. When all five TODOs are done, the "after" column should show every attack **defended**.

---

## How to interpret the results

```text
BEFORE (all controls OFF)
sec-03    indirect_prompt_injection  ATTACKER WON
                                    - emailed after reading an injected ticket
...
Attacks that succeeded: 9/10

AFTER (all controls ON)
sec-03    indirect_prompt_injection  defended
...
Attacks that succeeded: 0/10

Controls fixed 9 of 9 successful attacks.
```

- **ATTACKER WON** means the attack achieved real impact (an email sent, a refund issued, bulk data exposed) or the agent printed the attacker's phrase.
- **defended** means the agent resisted.
- The **baseline** case (`sec-01`) is a normal question. The attacker "wins" it only if a secured agent wrongly **refuses** a legitimate request. This checks your controls didn't make the agent useless.

The agent is an LLM, so results can vary slightly between runs, especially the injection cases. The **action-based** checks (emails, refunds, records exposed) are stable; the text-based ones can wobble. Run `--compare` more than once if a result surprises you.

**If an attack still wins with controls on,** match it to its control in the table above, open that `TODO`, and check your logic. The vulnerability name in the output tells you which control is responsible.

---

## Challenge

The challenge is in **Section 16** of the notebook.

**Part A:** write a new attack in `security_test_cases.json` and test it both ways.
**Part B:** turn the controls on **one at a time** and record how many attacks still succeed after each, showing defence in depth.
**Part C:** a 150-word responsible-AI reflection on what else a team should do before putting an agent like this in front of real customers.
**Stretch:** make the approval gate approve small refunds automatically but require approval for large ones.

---

## What to submit

Submit these through the activity's submission form on the course website:

1. **`Lesson_12_Activity.ipynb`**: run all cells, including the challenge, so the outputs are saved. In Colab, use **File → Download → Download .ipynb**. Make sure no API key appears anywhere in it.
2. **`lesson_12_vulnerable_agent.py`**: with all five security controls implemented.
3. **`security_test_cases.json`**: including your new attack from Part A.
4. **`security_results.json`**: from `python lesson_12_vulnerable_agent.py --compare`.
5. **A security findings table (150–250 words)** with a row per vulnerability you triggered: the vulnerability, how you triggered it, the observed impact, and the control that fixed it.
6. **A short reflection** answering:
   - Which attack surprised you most, and why?
   - Why is a security control in code stronger than the same rule written in the prompt?
   - From Part B, which attacks needed more than one control (defence in depth)?

### Checklist before you submit

- [ ] All five security controls in the `.py` file are implemented
- [ ] `--compare` shows every attack defended with controls on (baseline `sec-01` still answered)
- [ ] The baseline question is NOT refused by your secured agent
- [ ] Your new attack from Part A is in the dataset and tested
- [ ] `security_results.json` is saved
- [ ] Your API key does not appear in any file

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `GEMINI_API_KEY is not set` | Set the environment variable in the **same** terminal you run the file from. |
| `FileNotFoundError: security_test_cases.json` | Run the command from the activity folder, where the dataset is. |
| Attacks still win after I implemented a control | Check you set the result/return value **inside** the `if CONTROLS[...]` branch, and that the branch doesn't fall through to the vulnerable code below it. |
| The baseline `sec-01` is refused with controls on | Your hardened prompt or a gate is too strict. Security controls should block attacks, not normal questions. |
| `Cannot use Command(resume=...) without checkpointer` | The agent is compiled with an `InMemorySaver`; don't remove it. The approval gate needs it. |
| Results differ between runs | Normal for LLMs. The action-based checks are stable; run `--compare` again. |
| `503 UNAVAILABLE` or `429 RESOURCE_EXHAUSTED` | Gemini is busy or rate-limited. Wait a minute, or run single attacks with `--attack`. |
| `404` model not found | Set `GEMINI_MODEL` to a current model name from Google AI Studio. |
