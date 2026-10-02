"""
Lesson 7: Agentic Workflows
Activity 01: Multi-Step Support Workflow (starter file)

You will build a support-desk workflow for an online learning platform. A customer message goes
through several steps: an agent reads it, the workflow routes it to the right branch, looks up
information (in parallel), drafts a reply, reviews it in a loop, and asks a human to approve refunds.

    Input
      ↓
    Triage agent            (sequential step, structured output)
      ↓
    Route                   (conditional decision)
     ↙        ↘
  Billing     Technical      (branch A: parallel lookups + refund policy | branch B: help-doc search)
     ↘        ↙
    Draft reply
      ↓
    Review agent            (loop: review → revise, at most MAX_REVISIONS times)
      ↓
    Human approval          (human-in-the-loop, only for refunds)
      ↓
    Result

Most of the code is written for you. Complete the five TODO sections:

    TODO 1  route()                      conditional routing
    TODO 2  gather_billing_context()     parallel execution
    TODO 3  review_loop()                a review-and-revise loop
    TODO 4  human_approval()             human-in-the-loop
    TODO 5  run_workflow()               connect every step into one workflow

If you run the file before finishing a TODO, it stops with a message naming that TODO.

How to run (from this folder):

    python lesson_07_workflow.py --parallel-demo                       # compare sequential vs parallel (no API key needed)
    python lesson_07_workflow.py --run "Videos keep buffering" --customer C003   # run one message (you approve refunds)
    python lesson_07_workflow.py --test                                # run all test cases, save workflow_test_results.json

Set your API key first (see README.md):
    macOS / Linux:  export GEMINI_API_KEY="your-key"
    Windows:        set GEMINI_API_KEY=your-key
"""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from typing import Literal, Optional

from google import genai
from google.genai import types
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
MAX_REVISIONS = 2            # loop limit: how many times the reviewer can send a draft back
LOOKUP_DELAY = 1.0           # simulated database delay (seconds) for each lookup
REFUND_WINDOW_DAYS = 14      # refund policy: within 14 days of purchase...
MAX_PROGRESS_FOR_REFUND = 30 # ...and less than 30% of the course completed


# ---------------------------------------------------------------------------
# Mock data: a tiny "database" for the platform (provided)
# ---------------------------------------------------------------------------
TODAY = date.today()

CUSTOMERS = {
    "C001": {"name": "Lena Fischer", "plan": "Pro"},
    "C002": {"name": "Kenji Watanabe", "plan": "Basic"},
    "C003": {"name": "Maria Silva", "plan": "Pro"},
}

ORDERS = {
    "A-1001": {"customer_id": "C001", "course": "Python for Data Analysis", "price_usd": 49,
               "purchased": str(TODAY - timedelta(days=5)), "progress_pct": 10},
    "A-1002": {"customer_id": "C002", "course": "Intro to Machine Learning", "price_usd": 79,
               "purchased": str(TODAY - timedelta(days=40)), "progress_pct": 85},
}

HELP_DOCS = [
    {"title": "Videos won't play or keep buffering",
     "text": "Refresh the page, switch the video quality to 480p, and close other tabs that use the internet. "
             "If it still freezes, clear your browser cache or try a different browser.",
     "keywords": ["video", "buffering", "freezing", "loading", "playing", "stream", "slow", "lag"]},
    {"title": "Resetting your password",
     "text": "Click 'Forgot password' on the login page and follow the link we email you. The link expires after 1 hour.",
     "keywords": ["password", "login", "reset", "forgot", "locked", "sign"]},
    {"title": "Downloading your certificate",
     "text": "Certificates unlock when you finish 100% of a course. Open the course, choose 'Certificate', then 'Download PDF'.",
     "keywords": ["certificate", "download", "pdf", "complete", "finish", "proof"]},
    {"title": "Changing your account email",
     "text": "Go to Settings > Account > Email, enter the new address, and confirm it from the email we send you.",
     "keywords": ["email", "account", "change", "settings", "profile", "address"]},
    {"title": "Learning offline with the mobile app",
     "text": "In the mobile app, tap the download icon next to a lesson to watch it offline. Downloads last 30 days.",
     "keywords": ["mobile", "app", "offline", "phone", "download", "tablet", "android", "iphone"]},
]


# ---------------------------------------------------------------------------
# Part 1: State. One dictionary carries everything through the workflow.
# ---------------------------------------------------------------------------
def new_state(message: str, customer_id: str) -> dict:
    return {
        "message": message,          # input
        "customer_id": customer_id,
        "triage": None,              # set by triage()
        "route": None,               # set by route()
        "context": {},               # facts gathered by the branch
        "draft": None,               # the reply being worked on
        "reviews": [],               # every review from the loop
        "revisions": 0,
        "needs_approval": False,     # True for refunds
        "approval": None,            # the human's decision
        "status": "in_progress",     # becomes sent / escalated / held
        "final_reply": None,
        "trace": [],                 # a log of every step, for debugging
    }


def log_step(state: dict, step: str, started: float, note: str = "") -> None:
    """Record one step in the trace, with how long it took."""
    seconds = round(time.time() - started, 2)
    state["trace"].append({"step": step, "seconds": seconds, "note": note})
    print(f"  ✓ {step:<16} {seconds:>5.2f}s  {note}")


# ---------------------------------------------------------------------------
# Gemini helpers (provided)
# ---------------------------------------------------------------------------
_client = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit("GEMINI_API_KEY is not set. See 'API key setup' in README.md.")
        _client = genai.Client(api_key=api_key)
    return _client


def ask_llm(prompt: str, system: str, schema=None):
    """Call Gemini once. With a schema, returns a parsed object; otherwise returns text. Retries when busy."""
    config = types.GenerateContentConfig(system_instruction=system)
    if schema is not None:
        config.response_mime_type = "application/json"
        config.response_schema = schema
    for attempt in range(4):
        try:
            response = get_client().models.generate_content(model=MODEL, contents=prompt, config=config)
            return response.parsed if schema is not None else response.text.strip()
        except Exception as e:
            busy = any(code in str(e) for code in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"))
            if busy and attempt < 3:
                time.sleep(2 ** attempt)
            else:
                raise


# ---------------------------------------------------------------------------
# Step 1 (sequential): the triage agent reads the message
# ---------------------------------------------------------------------------
class Triage(BaseModel):
    category: Literal["billing", "technical"] = Field(
        description="billing = payments, refunds, charges, invoices. technical = anything about using the platform.")
    urgency: Literal["low", "medium", "high"]
    order_id: Optional[str] = Field(None, description="Order ID like A-1001 if the customer mentions one, else null")
    summary: str = Field(description="One sentence describing what the customer needs")


TRIAGE_SYSTEM = "You are the triage agent for an online learning platform's support desk. Read the customer message and classify it."


def triage(state: dict) -> dict:
    started = time.time()
    result = ask_llm(state["message"], TRIAGE_SYSTEM, schema=Triage)
    state["triage"] = result.model_dump()
    log_step(state, "triage", started, f"{result.category}, urgency {result.urgency}")
    return state


# ---------------------------------------------------------------------------
# Step 2 (conditional): route to a branch
# ---------------------------------------------------------------------------
def route(state: dict) -> str:
    """Decide which branch handles this message. Returns "billing" or "technical"."""
    started = time.time()
    category = state["triage"]["category"]

    # TODO 1: Decide the branch.
    #   - If category is "billing", branch = "billing". Otherwise branch = "technical".
    #   - Save the decision in the state: state["route"] = branch
    branch = None

    if branch is None:
        raise NotImplementedError("TODO 1: complete route()")
    log_step(state, "route", started, f"→ {branch} branch")
    return branch


# ---------------------------------------------------------------------------
# Branch A: billing (parallel lookups + a refund policy written in code)
# ---------------------------------------------------------------------------
def get_customer(customer_id: str) -> dict:
    time.sleep(LOOKUP_DELAY)   # pretend this is a slow database call
    return CUSTOMERS.get(customer_id, {"error": f"No customer {customer_id}"})


def get_order(order_id: Optional[str]) -> dict:
    time.sleep(LOOKUP_DELAY)   # pretend this is a slow database call
    if not order_id:
        return {"error": "No order ID given"}
    return ORDERS.get(order_id.upper(), {"error": f"No order {order_id}"})


def gather_billing_context(customer_id: str, order_id: Optional[str]) -> tuple[dict, dict]:
    """Look up the customer and the order AT THE SAME TIME. Returns (customer, order)."""
    # Doing them one after the other would take 2 x LOOKUP_DELAY:
    #     return get_customer(customer_id), get_order(order_id)
    #
    # TODO 2: Run both lookups in parallel with a ThreadPoolExecutor.
    #   with ThreadPoolExecutor() as executor:
    #       a) customer_job = executor.submit(get_customer, customer_id)
    #       b) order_job    = executor.submit(get_order, order_id)
    #       c) return customer_job.result(), order_job.result()
    #   submit() starts each job straight away; result() waits for it to finish.
    raise NotImplementedError("TODO 2: complete gather_billing_context()")


def check_refund_policy(order: dict) -> dict:
    """The refund rule lives in code, not in the LLM, so it's applied the same way every time."""
    if "error" in order:
        return {"eligible": False, "reason": "We couldn't find the order, so we need the order ID."}
    days = (TODAY - date.fromisoformat(order["purchased"])).days
    if days > REFUND_WINDOW_DAYS:
        return {"eligible": False, "reason": f"Purchased {days} days ago; refunds are only possible within {REFUND_WINDOW_DAYS} days."}
    if order["progress_pct"] >= MAX_PROGRESS_FOR_REFUND:
        return {"eligible": False, "reason": f"{order['progress_pct']}% of the course is completed; refunds need under {MAX_PROGRESS_FOR_REFUND}%."}
    return {"eligible": True, "reason": f"Purchased {days} days ago with {order['progress_pct']}% progress: within policy.",
            "amount_usd": order["price_usd"]}


WRITER_SYSTEM = """You write replies for an online learning platform's support desk.
Rules: greet the customer by first name, stay under 120 words, be warm and professional,
use ONLY the facts provided, never promise anything the facts don't support, and end with a clear next step.
Write only the reply text."""


def billing_branch(state: dict) -> dict:
    started = time.time()
    customer, order = gather_billing_context(state["customer_id"], state["triage"]["order_id"])
    policy = check_refund_policy(order)
    state["context"] = {"customer": customer, "order": order, "refund_policy": policy}
    state["needs_approval"] = policy["eligible"]   # a person approves every refund
    log_step(state, "billing lookups", started, f"refund eligible: {policy['eligible']}")

    started = time.time()
    prompt = (f"Customer message: {state['message']}\n\nFacts:\n{json.dumps(state['context'], indent=2)}\n\n"
              "If the refund is eligible, say the refund has been requested and will be confirmed shortly. "
              "If not, explain the reason kindly.")
    state["draft"] = ask_llm(prompt, WRITER_SYSTEM)
    log_step(state, "draft reply", started)
    return state


# ---------------------------------------------------------------------------
# Branch B: technical (search the help docs)
# ---------------------------------------------------------------------------
STOPWORDS = {"the", "and", "for", "with", "you", "your", "can", "what", "how", "my", "keep", "keeps",
             "on", "it", "is", "are", "do", "does", "i", "to", "a", "an", "of", "in", "course", "help", "please"}


def search_help_docs(text: str, min_matches: int = 2) -> list[dict]:
    """Return help docs that share at least `min_matches` keywords with the text, best first."""
    words = {w.rstrip("s") for w in re.findall(r"[a-z]+", text.lower()) if w not in STOPWORDS}
    scored = []
    for doc in HELP_DOCS:
        score = len(words & {k.rstrip("s") for k in doc["keywords"]})
        if score >= min_matches:
            scored.append((score, doc))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [doc for score, doc in scored]


def technical_branch(state: dict) -> dict:
    started = time.time()
    docs = search_help_docs(state["message"] + " " + state["triage"]["summary"])
    customer = CUSTOMERS.get(state["customer_id"], {})
    state["context"] = {"customer": customer, "help_docs": docs}
    log_step(state, "search help docs", started, f"{len(docs)} article(s) found")

    if not docs:
        # Conditional early exit: nothing to answer with, so a person takes over.
        first_name = customer.get("name", "there").split()[0]
        state["draft"] = (f"Hi {first_name}, thanks for your message. I've passed your question to our "
                          "technical team, and they'll reply within one working day.")
        state["status"] = "escalated"
        log_step(state, "escalate", time.time(), "no matching help article")
        return state

    started = time.time()
    prompt = f"Customer message: {state['message']}\n\nFacts:\n{json.dumps(state['context'], indent=2)}"
    state["draft"] = ask_llm(prompt, WRITER_SYSTEM)
    log_step(state, "draft reply", started)
    return state


# ---------------------------------------------------------------------------
# Step 3 (loop): the review agent checks the draft; revise until approved
# ---------------------------------------------------------------------------
class Review(BaseModel):
    approved: bool
    feedback: str = Field(description="If not approved: exactly what to fix. If approved: 'OK'.")


REVIEW_SYSTEM = """You are a strict quality reviewer for support replies. Approve the draft only if ALL are true:
1. It greets the customer by first name.
2. It is under 120 words.
3. Every claim matches the facts provided. It promises nothing the facts don't support.
4. It ends with a clear next step for the customer.
5. The tone is warm and professional."""


def review_draft(state: dict) -> Review:
    prompt = f"Facts:\n{json.dumps(state['context'], indent=2)}\n\nDraft reply:\n{state['draft']}"
    return ask_llm(prompt, REVIEW_SYSTEM, schema=Review)


def revise_draft(state: dict, feedback: str) -> str:
    prompt = (f"Customer message: {state['message']}\n\nFacts:\n{json.dumps(state['context'], indent=2)}\n\n"
              f"Current draft:\n{state['draft']}\n\nReviewer feedback:\n{feedback}\n\nRewrite the draft to fix the feedback.")
    return ask_llm(prompt, WRITER_SYSTEM)


def review_loop(state: dict) -> dict:
    """Review the draft; if it isn't approved, revise and review again, at most MAX_REVISIONS times."""
    while True:
        started = time.time()
        review = review_draft(state)
        state["reviews"].append(review.model_dump())
        log_step(state, "review", started, "approved" if review.approved else f"needs changes: {review.feedback[:60]}")

        # TODO 3: Finish the loop.
        #   a) If review.approved is True, stop the loop (break).
        #   b) If state["revisions"] has reached MAX_REVISIONS, stop too, but first:
        #        - set state["needs_approval"] = True   (a person must check a draft the reviewer never approved)
        #        - log it: log_step(state, "review limit", time.time(), "max revisions reached; a person will check it")
        #   c) Otherwise, revise and go round again:
        #        - state["draft"] = revise_draft(state, review.feedback)
        #        - add 1 to state["revisions"]
        #        - log it: log_step(state, "revise", started, f"revision {state['revisions']}")
        #      (set started = time.time() before calling revise_draft, so the log shows how long it took)
        raise NotImplementedError("TODO 3: complete the loop in review_loop()")
    return state


# ---------------------------------------------------------------------------
# Step 4 (human-in-the-loop): a person approves refunds before anything is sent
# ---------------------------------------------------------------------------
def console_approver(state: dict) -> dict:
    """Ask a real person in the terminal. Returns {"decision": "approve"|"edit"|"reject", "text": ...}."""
    print("\n" + "-" * 60)
    print("HUMAN APPROVAL NEEDED")
    policy = state["context"].get("refund_policy", {})
    if policy.get("eligible"):
        print(f"Refund: ${policy['amount_usd']}  ({policy['reason']})")
    print(f"\nDraft reply:\n{state['draft']}\n")
    choice = input("Approve (a), edit (e) or reject (r)? ").strip().lower()
    if choice.startswith("e"):
        return {"decision": "edit", "text": input("Type the new reply: ").strip()}
    if choice.startswith("r"):
        return {"decision": "reject", "text": None}
    return {"decision": "approve", "text": None}


def auto_approver(state: dict) -> dict:
    """Used by --test so the tests can run without anyone typing."""
    return {"decision": "approve", "text": None}


def human_approval(state: dict, approver=console_approver) -> dict:
    started = time.time()
    decision = approver(state)
    state["approval"] = decision["decision"]

    # TODO 4: Act on the human's decision.
    #   "approve" -> the draft is sent as it is:   state["final_reply"] = state["draft"]
    #   "edit"    -> send the human's version:     state["final_reply"] = decision["text"]
    #   "reject"  -> send nothing:                 state["final_reply"] = None  and  state["status"] = "held"
    raise NotImplementedError("TODO 4: complete human_approval()")

    log_step(state, "human approval", started, decision["decision"])
    return state


# ---------------------------------------------------------------------------
# Step 5: finish
# ---------------------------------------------------------------------------
def finalize(state: dict) -> dict:
    if state["status"] == "in_progress":
        state["status"] = "sent"
    if state["final_reply"] is None and state["status"] in ("sent", "escalated"):
        state["final_reply"] = state["draft"]
    return state


# ---------------------------------------------------------------------------
# The whole workflow
# ---------------------------------------------------------------------------
def run_workflow(message: str, customer_id: str, approver=console_approver) -> dict:
    print(f"\nCustomer {customer_id}: {message}")
    state = new_state(message, customer_id)

    # TODO 5: Connect the steps. Each step takes the state and returns it updated.
    #   1. state = triage(state)
    #   2. branch = route(state)
    #   3. If branch is "billing", run billing_branch(state); otherwise run technical_branch(state).
    #   4. If state["status"] is NOT "escalated":
    #        - run review_loop(state)
    #        - if state["needs_approval"] is True, run human_approval(state, approver)
    #      (Escalated messages skip review and approval and go straight to finalize.)
    raise NotImplementedError("TODO 5: connect the steps in run_workflow()")

    state = finalize(state)
    print(f"\n  Status: {state['status'].upper()}")
    if state["final_reply"]:
        print(f"  Reply:\n{state['final_reply']}")
    return state


# ---------------------------------------------------------------------------
# Demos and tests (provided)
# ---------------------------------------------------------------------------
def parallel_demo():
    print("Looking up a customer and an order, each with a 1-second simulated delay.\n")
    started = time.time()
    get_customer("C001")
    get_order("A-1001")
    print(f"One after the other: {time.time() - started:.2f}s")

    started = time.time()
    gather_billing_context("C001", "A-1001")
    print(f"In parallel:         {time.time() - started:.2f}s")


TEST_CASES = [
    {"name": "eligible refund", "customer_id": "C001",
     "message": "Hi, I bought Python for Data Analysis last week (order A-1001) but it isn't what I expected. Can I get a refund?",
     "expect": {"route": "billing", "needs_approval": True, "status": "sent"}},
    {"name": "refund outside policy", "customer_id": "C002",
     "message": "I want my money back for order A-1002. The course was too basic for me.",
     "expect": {"route": "billing", "needs_approval": False, "status": "sent"}},
    {"name": "help article found", "customer_id": "C003",
     "message": "The course videos keep buffering and freezing on my laptop. What can I do?",
     "expect": {"route": "technical", "status": "sent"}},
    {"name": "no help article: escalate", "customer_id": "C003",
     "message": "Can I connect my account to my smartwatch to track my study time?",
     "expect": {"route": "technical", "status": "escalated"}},
]


def run_tests(output_file: str = "workflow_test_results.json"):
    runs, checks = [], []
    for case in TEST_CASES:
        print("\n" + "=" * 60 + f"\nTEST: {case['name']}")
        state = run_workflow(case["message"], case["customer_id"], approver=auto_approver)
        runs.append({"test": case["name"], "state": state})
        for key, expected in case["expect"].items():
            actual = state.get(key)
            checks.append({"check": f"{case['name']}: {key} = {expected}", "ok": actual == expected, "actual": actual})
        checks.append({"check": f"{case['name']}: revisions within limit",
                       "ok": state["revisions"] <= MAX_REVISIONS, "actual": state["revisions"]})

    print("\n" + "=" * 60 + "\nCHECKS")
    for c in checks:
        print(f"  {'PASS ' if c['ok'] else 'CHECK'}  {c['check']}" + ("" if c["ok"] else f"  (got {c['actual']})"))
    print(f"\n{sum(c['ok'] for c in checks)}/{len(checks)} checks passed.")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"model": MODEL, "max_revisions": MAX_REVISIONS, "checks": checks, "runs": runs},
                  f, ensure_ascii=False, indent=2, default=str)
    print(f"Saved {output_file}. Submit this file with your work.")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lesson 7: a multi-step support workflow")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--parallel-demo", action="store_true", help="compare sequential and parallel lookups (no API key needed)")
    group.add_argument("--run", metavar="MESSAGE", help="run the workflow on one message")
    group.add_argument("--test", action="store_true", help="run all test cases")
    parser.add_argument("--customer", default="C001", help="customer ID for --run (C001, C002 or C003)")
    args = parser.parse_args()

    if args.parallel_demo:
        parallel_demo()
    elif args.run:
        run_workflow(args.run, args.customer)
    else:
        run_tests()
