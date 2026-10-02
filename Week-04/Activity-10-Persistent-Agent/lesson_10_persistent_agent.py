"""
Lesson 10: Agent State, Memory & Persistence
Activity 01: Persistent Agent (starter file)

You will build a Learning Plan agent that writes a personalised study plan, one
module at a time. Its progress is saved after every step, so if the program is
interrupted halfway through, it can recover and continue exactly where it stopped.

Two kinds of state are saved to two local files:

    checkpoints.db       CONVERSATION STATE: one thread per task. LangGraph saves a
                         checkpoint after every step (SqliteSaver).
    user_profiles.json   USER STATE: facts about each user that last across all
                         their conversations (a plain JSON file, like Lesson 6).

Most of the code is written for you. Complete the seven TODO sections:

    TODO 1  PlanState           define the agent's structured state
    TODO 2  get_checkpointer()  create a SQLite checkpointer (conversation persistence)
    TODO 3  save_profile()      save user state to the JSON file (user persistence)
    TODO 4  route_modules()     loop until every module is written
    TODO 5  build_graph()       add the edges and compile WITH the checkpointer
    TODO 6  start_task()        run the agent on a thread
    TODO 7  resume_task()       recover the saved state and continue

If you run the file before finishing a TODO, it stops with a message naming that TODO.

How to run (from this folder):

    python lesson_10_persistent_agent.py --graph                                  # diagram (no API key needed)
    python lesson_10_persistent_agent.py --start amara "Learn SQL for data analysis" --crash-at 3
    python lesson_10_persistent_agent.py --threads                                # list saved threads (no API key)
    python lesson_10_persistent_agent.py --status amara-1                         # look inside a saved thread
    python lesson_10_persistent_agent.py --resume amara-1                         # recover and continue
    python lesson_10_persistent_agent.py --ask amara-1 "Which module should I start with?"
    python lesson_10_persistent_agent.py --test                                   # run the full test

Set your API key first (see README.md):
    macOS / Linux:  export GEMINI_API_KEY="your-key"
    Windows:        set GEMINI_API_KEY=your-key
"""

import argparse
import json
import operator
import os
import sqlite3
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
CHECKPOINT_DB = "checkpoints.db"         # conversation state lives here
PROFILES_FILE = "user_profiles.json"     # user state lives here
NUM_MODULES = 4                          # modules in every learning plan
CRASH_AT_MODULE = None                   # set to a module number to simulate an interruption


class SimulatedCrash(Exception):
    """Raised on purpose to simulate the program being interrupted (a crash, a power cut, a closed laptop)."""


# ---------------------------------------------------------------------------
# The model (provided). Created only when first needed.
# ---------------------------------------------------------------------------
_llm = None


def get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
            raise SystemExit("GEMINI_API_KEY is not set. See 'API key setup' in README.md.")
        _llm = ChatGoogleGenerativeAI(model=MODEL)
    return _llm


# ---------------------------------------------------------------------------
# USER STATE: one JSON file, one entry per user
# {"amara": {"name": "Amara", "level": "beginner", "minutes_per_day": 30, "completed_plans": [...]}}
# ---------------------------------------------------------------------------
DEFAULT_PROFILE = {"level": "beginner", "minutes_per_day": 30, "completed_plans": []}


def load_profiles() -> dict:
    """Read every user profile. Returns an empty dict if the file doesn't exist yet."""
    if not os.path.exists(PROFILES_FILE):
        return {}
    with open(PROFILES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_profile(user_id: str) -> dict:
    """Return one user's profile, or a default profile for a new user."""
    return load_profiles().get(user_id, {"name": user_id.title(), **DEFAULT_PROFILE})


def save_profile(user_id: str, profile: dict) -> None:
    """Save one user's profile, keeping everyone else's."""
    # TODO 3: Load all profiles with load_profiles(), set profiles[user_id] = profile,
    #         then write the whole dictionary back to PROFILES_FILE with json.dump()
    #         (open the file with "w" and encoding="utf-8"; use indent=2 so it's easy to read).
    raise NotImplementedError("TODO 3: complete save_profile()")


# ---------------------------------------------------------------------------
# CONVERSATION STATE: the agent's structured state for one task (one thread)
# ---------------------------------------------------------------------------
class PlanState(TypedDict):
    messages: Annotated[list, add_messages]          # the conversation in this thread
    user_id: str                                     # who this thread belongs to

    # TODO 1: Add the other seven fields:
    #   profile         (dict)        a copy of the user's profile, loaded at the start
    #   goal            (str)         what the user wants to learn
    #   modules         (list[str])   the plan's module titles
    #   current_module  (int)         how many modules are written so far
    #   lessons         the written modules. Each node ADDS one, so give it a reducer:
    #                   Annotated[list[dict], operator.add]
    #   status          (str)         "planning", "writing" or "done"
    #   final_plan      (str)         the finished plan


REQUIRED_FIELDS = {"messages", "user_id", "profile", "goal", "modules", "current_module",
                   "lessons", "status", "final_plan"}


# ---------------------------------------------------------------------------
# Prompts and structured output (provided)
# ---------------------------------------------------------------------------
class Outline(BaseModel):
    """The plan's structure."""
    modules: list[str] = Field(description=f"Exactly {NUM_MODULES} short module titles, from first to last")


OUTLINE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You design short, practical learning plans. The learner is at {level} level and can study "
               "{minutes} minutes per day. Create exactly {num} modules that build on each other."),
    ("human", "Learning goal: {goal}"),
])

MODULE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You write one module of a learning plan for a {level} learner who studies {minutes} minutes "
               "per day. In under 90 words, give: what to learn, one hands-on practice task, and how many "
               "days it should take. Plain text, no headings."),
    ("human", "Learning goal: {goal}\nFull plan: {modules}\nWrite module {number}: {title}"),
])

CHAT_PROMPT = ("You are a friendly study coach. You already wrote this learning plan for {name}:\n\n{plan}\n\n"
               "Answer their questions about it briefly and encouragingly.")


# ---------------------------------------------------------------------------
# NODES
# ---------------------------------------------------------------------------
def load_profile(state: PlanState) -> dict:
    """Copy the user's long-term profile into this thread's state."""
    profile = get_profile(state["user_id"])
    print(f"  [load_profile] {profile['name']}: {profile['level']}, {profile['minutes_per_day']} min/day, "
          f"{len(profile['completed_plans'])} plan(s) completed before")
    return {"profile": profile, "status": "planning"}


def plan_outline(state: PlanState) -> dict:
    """Decide the plan's modules."""
    chain = OUTLINE_PROMPT | get_llm().with_structured_output(Outline)
    outline = chain.invoke({"goal": state["goal"], "level": state["profile"]["level"],
                            "minutes": state["profile"]["minutes_per_day"], "num": NUM_MODULES})
    modules = outline.modules[:NUM_MODULES]
    print(f"  [plan_outline] {modules}")
    return {"modules": modules, "current_module": 0, "status": "writing"}


def write_module(state: PlanState) -> dict:
    """Write the next module. This is the slow, multi-step part of the task."""
    number = state["current_module"] + 1
    title = state["modules"][state["current_module"]]

    if CRASH_AT_MODULE == number:
        raise SimulatedCrash(f"Interrupted while writing module {number}!")

    chain = MODULE_PROMPT | get_llm()
    reply = chain.invoke({"goal": state["goal"], "level": state["profile"]["level"],
                          "minutes": state["profile"]["minutes_per_day"],
                          "modules": ", ".join(state["modules"]), "number": number, "title": title})
    print(f"  [write_module] module {number}/{len(state['modules'])} written: {title}")
    return {"lessons": [{"title": title, "content": reply.text}], "current_module": number}


def finalize(state: PlanState) -> dict:
    """Assemble the plan, and record it in the user's long-term profile."""
    parts = [f"Learning plan: {state['goal']}\n"]
    for i, lesson in enumerate(state["lessons"], 1):
        parts.append(f"Module {i}: {lesson['title']}\n{lesson['content']}\n")
    final_plan = "\n".join(parts)

    profile = get_profile(state["user_id"])
    profile["completed_plans"] = profile["completed_plans"] + [state["goal"]]
    save_profile(state["user_id"], profile)
    print(f"  [finalize] plan saved; {profile['name']} has now completed {len(profile['completed_plans'])} plan(s)")

    return {"final_plan": final_plan, "status": "done",
            "messages": [AIMessage(content=f"Your {len(state['lessons'])}-module plan is ready!")]}


def chat(state: PlanState) -> dict:
    """Answer follow-up questions about a finished plan, using the saved conversation."""
    system = CHAT_PROMPT.format(name=state["profile"]["name"], plan=state["final_plan"])
    history = [m for m in state["messages"] if isinstance(m, (HumanMessage, AIMessage))]
    reply = get_llm().invoke([SystemMessage(content=system)] + history)
    return {"messages": [AIMessage(content=reply.text)]}


# ---------------------------------------------------------------------------
# ROUTING
# ---------------------------------------------------------------------------
def route_start(state: PlanState) -> str:
    """A finished plan means this is a follow-up question; otherwise start (or restart) the task."""
    return "chat" if state.get("status") == "done" else "load_profile"


def route_modules(state: PlanState) -> str:
    """Keep writing modules until every module in the plan is done."""
    # TODO 4: Return "write_module" if fewer modules are written than planned
    #         (compare state["current_module"] with len(state["modules"])), otherwise return "finalize".
    raise NotImplementedError("TODO 4: complete route_modules()")


# ---------------------------------------------------------------------------
# PERSISTENCE: the checkpointer
# ---------------------------------------------------------------------------
def get_checkpointer(db_path: str = CHECKPOINT_DB) -> SqliteSaver:
    """A checkpointer that saves every checkpoint to a SQLite file on disk."""
    # TODO 2: Open a SQLite connection to db_path and wrap it in a SqliteSaver.
    #   a) connection = sqlite3.connect(db_path, check_same_thread=False)
    #      (check_same_thread=False lets LangGraph use the connection safely)
    #   b) return SqliteSaver(connection)
    raise NotImplementedError("TODO 2: complete get_checkpointer()")


def build_graph(checkpointer):
    graph = StateGraph(PlanState)

    graph.add_node("load_profile", load_profile)
    graph.add_node("plan_outline", plan_outline)
    graph.add_node("write_module", write_module)
    graph.add_node("finalize", finalize)
    graph.add_node("chat", chat)

    missing = REQUIRED_FIELDS - set(PlanState.__annotations__)
    if missing:
        raise NotImplementedError(f"TODO 1: add these fields to PlanState: {sorted(missing)}")

    graph.add_conditional_edges(START, route_start, ["load_profile", "chat"])
    graph.add_edge("load_profile", "plan_outline")
    graph.add_edge("finalize", END)
    graph.add_edge("chat", END)

    # TODO 5a: Connect "plan_outline" to "write_module" with a normal edge.

    # TODO 5b: The module LOOP: a conditional edge from "write_module" that uses route_modules
    #          and can go to "write_module" (again) or "finalize".

    # TODO 5c: Compile WITH the checkpointer, so a checkpoint is saved after every step:
    #          change the next line to  graph.compile(checkpointer=checkpointer)
    app = graph.compile()

    _check_graph(app)
    return app


def _check_graph(app):
    """(Provided) Give a clear message if an edge or the checkpointer is missing."""
    edges = {(e.source, e.target) for e in app.get_graph().edges}
    if ("plan_outline", "write_module") not in edges:
        raise NotImplementedError("TODO 5a: connect plan_outline to write_module in build_graph()")
    if ("write_module", "write_module") not in edges or ("write_module", "finalize") not in edges:
        raise NotImplementedError("TODO 5b: add the module loop from write_module in build_graph()")
    if app.checkpointer is None:
        raise NotImplementedError("TODO 5c: compile the graph with the checkpointer in build_graph()")


# ---------------------------------------------------------------------------
# RUNNING, SAVING AND RECOVERING
# ---------------------------------------------------------------------------
def thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def start_task(app, user_id: str, goal: str, thread_id: str):
    """Start a new learning-plan task on a thread. Returns the final state, or None if interrupted."""
    print(f"\nStarting task on thread '{thread_id}' for user '{user_id}': {goal}")
    first_state = {"messages": [HumanMessage(content=f"Make me a learning plan: {goal}")],
                   "user_id": user_id, "goal": goal, "lessons": []}
    try:
        # TODO 6: Run the graph on this thread and return the final state:
        #         app.invoke(first_state, thread_config(thread_id))
        #         The thread id in the config is what makes the checkpoints belong to THIS task.
        raise NotImplementedError("TODO 6: run the graph in start_task()")
    except SimulatedCrash as crash:
        print(f"\n  INTERRUPTED: {crash}")
        print(f"  Progress up to the last finished step is saved. Resume thread '{thread_id}' to continue.")
        return None


def resume_task(app, thread_id: str):
    """Recover a thread's saved state and continue from where it stopped."""
    config = thread_config(thread_id)

    # TODO 7a: Load the latest saved checkpoint for this thread:  saved = app.get_state(config)
    saved = None

    if saved is None:
        raise NotImplementedError("TODO 7a: load the saved state in resume_task()")
    if not saved.values:
        print(f"No saved state for thread '{thread_id}'.")
        return None
    if not saved.next:
        print(f"Thread '{thread_id}' has nothing left to do (status: {saved.values.get('status')}).")
        return saved.values

    print(f"\nRecovered thread '{thread_id}': {saved.values.get('current_module', 0)} of "
          f"{len(saved.values.get('modules', []))} modules done. Next step: {saved.next[0]}")
    try:
        # TODO 7b: Continue from the checkpoint. Invoke the graph with None as the input
        #          (meaning "no new input, just carry on") and the same config, and return the result.
        raise NotImplementedError("TODO 7b: continue the task in resume_task()")
    except SimulatedCrash as crash:
        print(f"\n  INTERRUPTED again: {crash}")
        return None


def ask(app, thread_id: str, question: str) -> str:
    """Ask a follow-up question in an existing thread."""
    result = app.invoke({"messages": [HumanMessage(content=question)]}, thread_config(thread_id))
    return result["messages"][-1].text


def show_status(app, thread_id: str) -> None:
    """Print what is saved for a thread, and its checkpoint history."""
    config = thread_config(thread_id)
    saved = app.get_state(config)
    if not saved.values:
        print(f"No saved state for thread '{thread_id}'.")
        return
    v = saved.values
    print(f"\nThread '{thread_id}'")
    print(f"  user:            {v.get('user_id')}")
    print(f"  goal:            {v.get('goal')}")
    print(f"  status:          {v.get('status')}")
    print(f"  modules written: {v.get('current_module', 0)} / {len(v.get('modules', []))}")
    print(f"  messages:        {len(v.get('messages', []))}")
    print(f"  next step:       {saved.next[0] if saved.next else '(finished)'}")
    history = list(app.get_state_history(config))
    print(f"  checkpoints:     {len(history)} saved (newest first):")
    for snapshot in history[:8]:
        nxt = snapshot.next[0] if snapshot.next else "-"
        print(f"    step {snapshot.metadata.get('step'):>3}  next: {nxt:13} modules done: "
              f"{snapshot.values.get('current_module', 0)}")


def list_threads(db_path: str = CHECKPOINT_DB) -> list[tuple]:
    """List every thread saved in the SQLite file, with its number of checkpoints."""
    if not os.path.exists(db_path):
        return []
    with sqlite3.connect(db_path) as conn:
        try:
            return conn.execute(
                "SELECT thread_id, COUNT(*) FROM checkpoints GROUP BY thread_id ORDER BY thread_id").fetchall()
        except sqlite3.OperationalError:   # the file exists but nothing has been saved yet
            return []


# ---------------------------------------------------------------------------
# Test (provided)
# ---------------------------------------------------------------------------
def run_tests(output_file: str = "persistence_test_results.json"):
    global CHECKPOINT_DB, PROFILES_FILE, CRASH_AT_MODULE
    CHECKPOINT_DB, PROFILES_FILE = "test_checkpoints.db", "test_user_profiles.json"
    for path in (CHECKPOINT_DB, PROFILES_FILE):
        if os.path.exists(path):
            os.remove(path)
    save_profile("amara", {"name": "Amara", "level": "beginner", "minutes_per_day": 30, "completed_plans": []})
    save_profile("ben", {"name": "Ben", "level": "intermediate", "minutes_per_day": 60, "completed_plans": []})
    checks = []

    # 1. Amara's task is interrupted while writing module 3.
    print("\n=== 1. Run Amara's task and interrupt it at module 3 ===")
    CRASH_AT_MODULE = 3
    app = build_graph(get_checkpointer(CHECKPOINT_DB))
    result = start_task(app, "amara", "Learn SQL for data analysis", "amara-1")
    saved = app.get_state(thread_config("amara-1"))
    checks.append({"check": "The task was interrupted", "ok": result is None})
    checks.append({"check": "2 modules were saved before the interruption",
                   "ok": saved.values.get("current_module") == 2 and len(saved.values.get("lessons", [])) == 2})
    checks.append({"check": "The saved state knows the next step is write_module", "ok": saved.next == ("write_module",)})

    # 2. "Restart the program": throw away the app and connection, then rebuild from the file on disk.
    print("\n=== 2. Restart: build a brand-new app from the SQLite file and resume ===")
    CRASH_AT_MODULE = None
    app.checkpointer.conn.close()
    app = build_graph(get_checkpointer(CHECKPOINT_DB))
    result = resume_task(app, "amara-1")
    checks.append({"check": "Resuming finished the plan", "ok": bool(result) and result.get("status") == "done"})
    checks.append({"check": f"The finished plan has exactly {NUM_MODULES} modules (none repeated or lost)",
                   "ok": bool(result) and len(result.get("lessons", [])) == NUM_MODULES})

    # 3. A second user, with their own thread and profile.
    print("\n=== 3. A second user: Ben ===")
    result_ben = start_task(app, "ben", "Build a personal website with HTML and CSS", "ben-1")
    checks.append({"check": "Ben's plan finished without interruption", "ok": bool(result_ben) and result_ben["status"] == "done"})
    checks.append({"check": "Ben's thread used Ben's profile (intermediate, 60 min/day)",
                   "ok": bool(result_ben) and result_ben["profile"]["level"] == "intermediate"})
    checks.append({"check": "Amara's and Ben's threads are separate",
                   "ok": app.get_state(thread_config("amara-1")).values["goal"] != result_ben["goal"]})

    # 4. Follow-up question in Amara's saved thread.
    print("\n=== 4. Amara comes back later with a follow-up question ===")
    reply = ask(app, "amara-1", "I have a busy week. Which module should I start with?")
    print(f"  Coach: {reply}")
    amara_messages = app.get_state(thread_config("amara-1")).values["messages"]
    checks.append({"check": "The follow-up was added to Amara's saved conversation", "ok": len(amara_messages) >= 4})

    # 5. User state was updated in the JSON file.
    profiles = load_profiles()
    checks.append({"check": "Amara's profile records 1 completed plan", "ok": len(profiles["amara"]["completed_plans"]) == 1})
    checks.append({"check": "Ben's profile records 1 completed plan", "ok": len(profiles["ben"]["completed_plans"]) == 1})
    threads = dict(list_threads(CHECKPOINT_DB))
    checks.append({"check": "Both threads are stored in the SQLite file", "ok": {"amara-1", "ben-1"} <= set(threads)})

    print("\n=== Checks ===")
    for c in checks:
        print(f"  {'PASS' if c['ok'] else 'CHECK'}  {c['check']}")
    print(f"\n{sum(c['ok'] for c in checks)}/{len(checks)} checks passed.")

    show_status(app, "amara-1")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"model": MODEL, "checks": checks, "threads": threads, "profiles": profiles,
                   "amara_plan": app.get_state(thread_config("amara-1")).values.get("final_plan", ""),
                   "follow_up_reply": reply}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {output_file}. Submit this file with your work.")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lesson 10: a persistent learning-plan agent")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--graph", action="store_true", help="print the graph as a Mermaid diagram (no API key needed)")
    group.add_argument("--start", nargs=2, metavar=("USER", "GOAL"), help="start a new task for a user")
    group.add_argument("--resume", metavar="THREAD", help="recover a thread and continue it")
    group.add_argument("--status", metavar="THREAD", help="show what is saved for a thread (no API key needed)")
    group.add_argument("--threads", action="store_true", help="list saved threads (no API key needed)")
    group.add_argument("--ask", nargs=2, metavar=("THREAD", "QUESTION"), help="ask a follow-up question in a thread")
    group.add_argument("--test", action="store_true", help="run the full test")
    parser.add_argument("--thread", help="thread id for --start (default: USER-1, USER-2, ...)")
    parser.add_argument("--crash-at", type=int, help="simulate an interruption while writing this module number")
    args = parser.parse_args()

    if args.threads:
        rows = list_threads()
        print("\n".join(f"  {t}  ({n} checkpoints)" for t, n in rows) if rows else "No saved threads yet.")
    elif args.test or not any([args.graph, args.start, args.resume, args.status, args.ask]):
        run_tests()
    else:
        app = build_graph(get_checkpointer())
        if args.graph:
            print(app.get_graph().draw_mermaid())
            print("Paste this into https://mermaid.live to see the diagram.")
        elif args.status:
            show_status(app, args.status)
        elif args.start:
            user_id, goal = args.start
            existing = {t for t, _ in list_threads()}
            thread_id = args.thread or next(f"{user_id}-{n}" for n in range(1, 1000) if f"{user_id}-{n}" not in existing)
            CRASH_AT_MODULE = args.crash_at
            result = start_task(app, user_id.lower(), goal, thread_id)
            if result:
                print("\n" + result["final_plan"])
        elif args.resume:
            result = resume_task(app, args.resume)
            if result and result.get("final_plan"):
                print("\n" + result["final_plan"])
        elif args.ask:
            print("\nCoach:", ask(app, *args.ask))
