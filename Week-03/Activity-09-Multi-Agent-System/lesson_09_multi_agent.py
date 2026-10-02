"""
Lesson 9: Multi-Agent Systems
Activity 04: Research Team Agent (starter file)

You will build a small research team: four specialised agents and a supervisor
that coordinates them. They collaborate on one research question through a
shared state, like a team writing on the same whiteboard.

    Researcher  gathers information with tools (course notes + Wikipedia)
    Analyst     organises the findings into key points, gaps and an outline
    Writer      writes a short report from the analysis
    Reviewer    checks the report and approves it or asks for changes
    Supervisor  decides which agent works next, and when the team is done

Most of the code is written for you. Complete the seven TODO sections:

    TODO 1  ResearchState        the shared state every agent reads and writes
    TODO 2  researcher()         run the research agent and save its notes
    TODO 3  analyst()            organise the notes with structured output
    TODO 4  writer()             write (or revise) the report
    TODO 5  reviewer()           check the report and give feedback
    TODO 6  supervisor()         decide which agent works next
    TODO 7  build_graph()        connect the supervisor and the agents

If you run the file before finishing a TODO, it stops with a message naming that TODO.

How to run (from this folder):

    python lesson_09_multi_agent.py --graph                    # print the graph diagram (no API key needed)
    python lesson_09_multi_agent.py --check-supervisor         # test the supervisor's rules (no API key needed)
    python lesson_09_multi_agent.py --topic "How do AI agents use memory?"   # research one topic
    python lesson_09_multi_agent.py --test                     # run the full test, save research_team_results.json

Set your API key first (see README.md):
    macOS / Linux:  export GEMINI_API_KEY="your-key"
    Windows:        set GEMINI_API_KEY=your-key
"""

import argparse
import json
import operator
import os
from typing import Annotated, TypedDict

import requests
from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
EMBEDDING_MODEL = "gemini-embedding-001"
MAX_REVISIONS = 2      # the Writer may revise at most this many times after the first draft
MAX_WORDS = 250        # target length for the report


# ---------------------------------------------------------------------------
# The course notes: the same 8 documents as Lessons 5 and 8
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE = [
    {
        "id": "kb-01",
        "title": "What is an AI agent?",
        "text": (
            "An AI agent is a system that uses a language model to pursue a goal by deciding what to do next. "
            "Unlike a chatbot, which only replies to messages, an agent can take actions with tools, look at the "
            "results, and choose its next step. A workflow follows a fixed path written by a developer; an agent "
            "chooses its own path. Most real agents are assisted: a human approves important actions such as "
            "payments or sending emails."
        ),
    },
    {
        "id": "kb-02",
        "title": "Tools and function calling",
        "text": (
            "A tool is a function an agent can ask to use, such as a calculator, a web search or an API. The model "
            "never runs the tool itself. It returns a function call containing the tool name and arguments; your "
            "Python code runs the function and sends the result back. Each tool is described with a schema: a name, "
            "a description and parameters. The model chooses tools using only these descriptions, so clear "
            "descriptions matter more than clever code."
        ),
    },
    {
        "id": "kb-03",
        "title": "The ReAct loop and stopping rules",
        "text": (
            "ReAct stands for Reason and Act. The agent reasons about what it needs, acts by calling a tool, "
            "observes the result, and repeats until it can answer. Every agent loop needs three stopping rules: "
            "stop when the model makes no more tool calls, stop when the step budget runs out, and block repeated "
            "identical calls with a loop guard. Without them, an agent can loop forever and waste money."
        ),
    },
    {
        "id": "kb-04",
        "title": "Agent memory",
        "text": (
            "Short-term memory is the conversation history the agent sends to the model on every turn. It is limited "
            "by the context window, the maximum number of tokens the model can read at once. Long-term memory lives "
            "outside the model, for example in a database, and is retrieved only when needed. Retrieval lets an "
            "agent use far more information than fits in its context window."
        ),
    },
    {
        "id": "kb-05",
        "title": "Retrieval-Augmented Generation (RAG)",
        "text": (
            "RAG has three steps: retrieve relevant documents, augment the prompt with them, and generate an answer "
            "grounded in that text. It reduces hallucination because the model answers from real sources it can "
            "cite. Unlike fine-tuning, RAG does not change the model's weights; to update its knowledge you simply "
            "update the documents. In agentic RAG, retrieval is a tool, and the agent decides when to search and "
            "what to search for."
        ),
    },
    {
        "id": "kb-06",
        "title": "Embeddings and vector databases",
        "text": (
            "An embedding is a list of numbers that represents the meaning of a piece of text. Texts with similar "
            "meanings have embeddings that are close together, even if they share no words, so 'car' and 'vehicle' "
            "end up near each other. Closeness is usually measured with cosine similarity. A vector database such "
            "as ChromaDB stores embeddings and quickly finds the ones closest to a query. This is called semantic search."
        ),
    },
    {
        "id": "kb-07",
        "title": "Multi-agent systems",
        "text": (
            "A multi-agent system splits work between several agents with different roles, such as a planner, a "
            "researcher and a writer. An orchestrator agent assigns tasks and combines the results. Agents pass work "
            "to each other through handoffs. Multiple agents help with large or varied tasks, but they cost more and "
            "are harder to debug, so start with a single agent and add more only when you need them."
        ),
    },
    {
        "id": "kb-08",
        "title": "Agent safety and guardrails",
        "text": (
            "Agents act in the real world, so they need guardrails. Give each agent only the tools it needs, a "
            "principle called least privilege. Ask a human to approve risky actions such as payments, deletions or "
            "sending messages. Treat text that comes back from tools and documents as data, never as instructions; "
            "attackers can hide commands in web pages, an attack called prompt injection. Log every step so you can "
            "audit what the agent did."
        ),
    },
]

# ---------------------------------------------------------------------------
# LangChain components (provided)
# Created only when first needed, so --graph and --check-supervisor work without an API key.
# ---------------------------------------------------------------------------
_llm = None
_retriever = None


def get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
            raise SystemExit("GEMINI_API_KEY is not set. See 'API key setup' in README.md.")
        _llm = ChatGoogleGenerativeAI(model=MODEL)
    return _llm


def get_retriever():
    global _retriever
    if _retriever is None:
        documents = [Document(page_content=d["text"], metadata={"title": d["title"]}) for d in KNOWLEDGE_BASE]
        store = InMemoryVectorStore.from_documents(documents, GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL))
        _retriever = store.as_retriever(search_kwargs={"k": 2})
    return _retriever


# ---------------------------------------------------------------------------
# The Researcher's tools (provided)
# ---------------------------------------------------------------------------
@tool
def search_course_notes(query: str) -> str:
    """Search the course notes about agentic AI: agents, tools, the ReAct loop, memory, RAG,
    embeddings, multi-agent systems and agent safety. Returns the most relevant notes with their titles."""
    try:
        docs = get_retriever().invoke(query)
        return "\n\n".join(f"[Course notes: {d.metadata['title']}]\n{d.page_content}" for d in docs)
    except Exception as e:
        return f"Error: course notes search failed: {e}"


@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for background information on any topic. Returns a short summary and the article title."""
    headers = {"User-Agent": "AgenticAICourse/1.0 (educational lab)"}
    try:
        search = requests.get("https://en.wikipedia.org/w/api.php", headers=headers, timeout=10, params={
            "action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 1}).json()
        hits = search.get("query", {}).get("search", [])
        if not hits:
            return f"No Wikipedia article found for '{query}'."
        title = hits[0]["title"]
        page = requests.get("https://en.wikipedia.org/api/rest_v1/page/summary/"
                            + requests.utils.quote(title.replace(" ", "_")), headers=headers, timeout=10).json()
        return f"[Wikipedia: {title}]\n{page.get('extract', '')[:1200]}"
    except Exception as e:
        return f"Error: Wikipedia search failed: {e}"


RESEARCH_TOOLS = [search_course_notes, search_wikipedia]


# ---------------------------------------------------------------------------
# Instructions and output formats for each agent (provided)
# ---------------------------------------------------------------------------
RESEARCHER_PROMPT = """You are the Researcher on a research team. Gather facts for the research question.
- Search the course notes first, then Wikipedia for wider background. Make 2 to 4 searches in total.
- Reply with 4 to 8 bullet points. Each bullet is one fact, followed by its source in brackets,
  for example: (Source: Course notes: Agent memory) or (Source: Wikipedia: Retrieval-augmented generation).
- Only include facts you found with the tools. Don't write the report; that's the Writer's job."""


class Analysis(BaseModel):
    """The Analyst's output."""
    key_points: list[str] = Field(description="The 3-5 most important points that answer the question")
    gaps: list[str] = Field(description="Anything important the notes don't cover (can be empty)")
    outline: list[str] = Field(description="3-4 section headings for the report, in order")


ANALYST_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are the Analyst on a research team. Organise the Researcher's notes. "
               "Use only the notes; don't add new facts."),
    ("human", "Research question: {topic}\n\nResearcher's notes:\n{notes}"),
])

WRITER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are the Writer on a research team. Write a short report for beginners, in Markdown, "
               "under {max_words} words. Follow the outline as section headings. Use only facts from the notes. "
               "End with a 'Sources' section listing the sources you used."),
    ("human", "Research question: {topic}\n\nOutline: {outline}\n\nKey points:\n{key_points}\n\n"
              "Researcher's notes:\n{notes}\n\nReviewer feedback to fix (if any): {feedback}"),
])


class Review(BaseModel):
    """The Reviewer's output."""
    approved: bool = Field(description="True only if the report meets every criterion")
    score: int = Field(description="Overall quality from 1 (poor) to 10 (excellent)")
    feedback: str = Field(description="Specific changes the Writer should make; empty if approved")


REVIEWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are the Reviewer on a research team. Check the report against these criteria:\n"
               "1. It answers the research question.\n"
               "2. Every fact is supported by the Researcher's notes (no invented facts).\n"
               "3. It ends with a Sources section.\n"
               "4. It is under {max_words} words and easy for a beginner to understand.\n"
               "Approve only if all four are met. Otherwise give short, specific feedback."),
    ("human", "Research question: {topic}\n\nResearcher's notes:\n{notes}\n\nReport to review:\n{draft}"),
])


# ---------------------------------------------------------------------------
# SHARED STATE
# Every agent reads from and writes to this one dictionary.
# ---------------------------------------------------------------------------
class ResearchState(TypedDict):
    topic: str                                     # the research question
    team_log: Annotated[list[str], operator.add]   # every agent adds a line: the team's message board

    # TODO 1: Add the other seven fields. Each one is written by a different team member:
    #   notes         (str)   Researcher -> facts with sources
    #   analysis      (dict)  Analyst    -> key_points, gaps, outline
    #   draft         (str)   Writer     -> the current report
    #   revision      (int)   how many drafts the Writer has written
    #   review        (dict)  Reviewer   -> approved, score, feedback, draft_number
    #   next_agent    (str)   Supervisor -> who works next
    #   final_report  (str)   the approved (or last) report


REQUIRED_FIELDS = {"topic", "notes", "analysis", "draft", "revision", "review", "next_agent", "final_report", "team_log"}


# ---------------------------------------------------------------------------
# THE AGENTS
# Each agent is a node: it reads the shared state and returns only what it changes.
# ---------------------------------------------------------------------------
_research_agent = None


def get_research_agent():
    """A LangChain tool-using agent (the loop from Lesson 4, prebuilt)."""
    global _research_agent
    if _research_agent is None:
        _research_agent = create_agent(model=get_llm(), tools=RESEARCH_TOOLS, system_prompt=RESEARCHER_PROMPT)
    return _research_agent


def researcher(state: ResearchState) -> dict:
    """Gather facts about the topic using tools."""
    # TODO 2a: Run the research agent. Invoke get_research_agent() with:
    #              {"messages": [HumanMessage(content=f"Research question: {state['topic']}")]}
    #          The agent's final answer (its notes) is the .text of the LAST message in result["messages"].
    result = None
    notes = None

    if result is None or notes is None:
        raise NotImplementedError("TODO 2a: run the research agent in researcher()")
    searches = sum(len(getattr(m, "tool_calls", None) or []) for m in result["messages"])
    print(f"  [Researcher] made {searches} searches")

    # TODO 2b: Return the updates: "notes", and a "team_log" list with ONE line, e.g.
    #          [f"Researcher: gathered notes using {searches} searches."]
    raise NotImplementedError("TODO 2b: return the notes from researcher()")


def analyst(state: ResearchState) -> dict:
    """Organise the notes into key points, gaps and an outline."""
    # TODO 3a: Build a chain that sends ANALYST_PROMPT to the model and returns an Analysis object:
    #              ANALYST_PROMPT | get_llm().with_structured_output(Analysis)
    #          Then invoke it with the "topic" and "notes" from the state.
    analysis = None

    if analysis is None:
        raise NotImplementedError("TODO 3a: run the analyst chain in analyst()")
    print(f"  [Analyst] {len(analysis.key_points)} key points, outline: {analysis.outline}")

    # TODO 3b: Return the updates: "analysis" as a dictionary (hint: analysis.model_dump()),
    #          and a one-line "team_log", e.g. f"Analyst: found {len(analysis.key_points)} key points."
    raise NotImplementedError("TODO 3b: return the analysis from analyst()")


def writer(state: ResearchState) -> dict:
    """Write the report, or revise it using the Reviewer's feedback."""
    feedback = state["review"].get("feedback", "") if state.get("review") else ""
    chain = WRITER_PROMPT | get_llm()
    reply = chain.invoke({
        "topic": state["topic"],
        "max_words": MAX_WORDS,
        "outline": ", ".join(state["analysis"]["outline"]),
        "key_points": "\n".join(f"- {p}" for p in state["analysis"]["key_points"]),
        "notes": state["notes"],
        "feedback": feedback or "none",
    })

    # TODO 4: Count this draft and return the updates.
    #   a) revision = the previous revision number + 1   (hint: state.get("revision", 0) + 1)
    #   b) Return "draft" (the reply's .text), "revision", and a one-line "team_log",
    #      e.g. "Writer: wrote the first draft." or "Writer: revised the report (draft 2)."
    raise NotImplementedError("TODO 4: count the draft and return it from writer()")


def reviewer(state: ResearchState) -> dict:
    """Check the report against the criteria and approve it or ask for changes."""
    chain = REVIEWER_PROMPT | get_llm().with_structured_output(Review)
    review = chain.invoke({"topic": state["topic"], "notes": state["notes"],
                           "draft": state["draft"], "max_words": MAX_WORDS})
    verdict = "approved" if review.approved else "asked for changes"
    print(f"  [Reviewer] {verdict} (score {review.score}/10){'' if review.approved else ': ' + review.feedback}")

    # TODO 5: Return the updates:
    #   "review"   -> the review as a dictionary, PLUS the number of the draft it reviewed:
    #                 {**review.model_dump(), "draft_number": state["revision"]}
    #                 (the supervisor uses draft_number to know whether the latest draft has been reviewed)
    #   "team_log" -> one line, e.g. f"Reviewer: {verdict} (draft {state['revision']}, score {review.score}/10)."
    raise NotImplementedError("TODO 5: return the review from reviewer()")


def finish(state: ResearchState) -> dict:
    """Publish the last draft as the final report."""
    approved = state.get("review", {}).get("approved", False)
    note = "" if approved else "\n\n_Note: the revision limit was reached before the Reviewer approved this report._"
    return {"final_report": state["draft"] + note,
            "team_log": [f"Supervisor: finished ({'approved' if approved else 'revision limit reached'})."]}


# ---------------------------------------------------------------------------
# THE SUPERVISOR
# It doesn't research or write. It only looks at the shared state and delegates.
# ---------------------------------------------------------------------------
def supervisor(state: ResearchState) -> dict:
    """Decide which agent should work next."""
    review = state.get("review") or {}
    revision = state.get("revision", 0)

    if not state.get("notes"):
        next_agent = "researcher"

    # TODO 6: Add the remaining rules as elif branches, in this order:
    #   - no analysis yet                                  -> "analyst"
    #   - no draft yet                                     -> "writer"
    #   - review.get("draft_number") != revision           -> "reviewer"   (latest draft not reviewed yet)
    #   - review.get("approved") is true                   -> "finish"
    #   - revision <= MAX_REVISIONS                        -> "writer"     (revise using the feedback)
    #   - otherwise                                        -> "finish"     (revision limit reached)
    #   Then delete the else branch below.
    else:
        raise NotImplementedError("TODO 6: add the remaining rules to supervisor()")

    print(f"[Supervisor] -> {next_agent}")
    return {"next_agent": next_agent}


def route_to_agent(state: ResearchState) -> str:
    """Routing function: follow the supervisor's decision."""
    return state["next_agent"]


# ---------------------------------------------------------------------------
# THE WORKFLOW: supervisor in the middle, every agent reports back to it
# ---------------------------------------------------------------------------
AGENTS = ["researcher", "analyst", "writer", "reviewer"]


def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("supervisor", supervisor)
    graph.add_node("researcher", researcher)
    graph.add_node("analyst", analyst)
    graph.add_node("writer", writer)
    graph.add_node("reviewer", reviewer)
    graph.add_node("finish", finish)

    missing = REQUIRED_FIELDS - set(ResearchState.__annotations__)
    if missing:
        raise NotImplementedError(f"TODO 1: add these fields to ResearchState: {sorted(missing)}")

    graph.add_edge(START, "supervisor")
    graph.add_edge("finish", END)

    # TODO 7a: Add a CONDITIONAL edge from "supervisor". It uses route_to_agent to choose,
    #          and can go to any agent or to "finish":
    #              graph.add_conditional_edges("supervisor", route_to_agent, AGENTS + ["finish"])

    # TODO 7b: Every agent reports back to the supervisor. Add a normal edge from each
    #          agent in AGENTS to "supervisor" (hint: a for loop).

    app = graph.compile()
    _check_graph(app)
    return app


def _check_graph(app):
    """(Provided) Give a clear message if an edge is missing."""
    edges = {(e.source, e.target) for e in app.get_graph().edges}
    for agent in AGENTS + ["finish"]:
        if ("supervisor", agent) not in edges:
            raise NotImplementedError(f"TODO 7a: the supervisor can't reach '{agent}' in build_graph()")
    for agent in AGENTS:
        if (agent, "supervisor") not in edges:
            raise NotImplementedError(f"TODO 7b: '{agent}' doesn't report back to the supervisor in build_graph()")


# ---------------------------------------------------------------------------
# Running the team (provided)
# ---------------------------------------------------------------------------
def new_task(topic: str) -> dict:
    return {"topic": topic, "notes": "", "analysis": {}, "draft": "", "revision": 0,
            "review": {}, "next_agent": "", "final_report": "", "team_log": []}


def run_team(topic: str, app=None) -> dict:
    app = app or build_graph()
    print(f"\nResearch question: {topic}\n")
    result = app.invoke(new_task(topic), {"recursion_limit": 40})
    print("\n--- Team log ---")
    for line in result["team_log"]:
        print(" ", line)
    print("\n--- Final report ---\n")
    print(result["final_report"])
    return result


SUPERVISOR_CASES = [
    ("Nothing done yet", {}, "researcher"),
    ("Notes gathered", {"notes": "..."}, "analyst"),
    ("Analysis done", {"notes": "...", "analysis": {"outline": []}}, "writer"),
    ("First draft written", {"notes": "...", "analysis": {"x": 1}, "draft": "...", "revision": 1}, "reviewer"),
    ("Draft 1 rejected", {"notes": "...", "analysis": {"x": 1}, "draft": "...", "revision": 1,
                          "review": {"approved": False, "draft_number": 1}}, "writer"),
    ("Draft 2 not reviewed yet", {"notes": "...", "analysis": {"x": 1}, "draft": "...", "revision": 2,
                                  "review": {"approved": False, "draft_number": 1}}, "reviewer"),
    ("Draft 2 approved", {"notes": "...", "analysis": {"x": 1}, "draft": "...", "revision": 2,
                          "review": {"approved": True, "draft_number": 2}}, "finish"),
    ("Revision limit reached", {"notes": "...", "analysis": {"x": 1}, "draft": "...", "revision": MAX_REVISIONS + 1,
                                "review": {"approved": False, "draft_number": MAX_REVISIONS + 1}}, "finish"),
]


def check_supervisor() -> list[dict]:
    """Test the supervisor's rules with hand-made states. No API calls."""
    print("Checking the supervisor's decisions:\n")
    results = []
    for name, partial_state, expected in SUPERVISOR_CASES:
        state = {**new_task("test"), **partial_state}
        got = supervisor(state)["next_agent"]
        ok = got == expected
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: expected {expected}, got {got}\n")
        results.append({"case": name, "expected": expected, "got": got, "ok": ok})
    print(f"{sum(r['ok'] for r in results)}/{len(results)} supervisor checks passed.")
    return results


TEST_TOPICS = [
    "How do AI agents use memory?",
    "What is retrieval-augmented generation, and why does it reduce hallucinations?",
]


def run_tests(output_file: str = "research_team_results.json"):
    supervisor_checks = check_supervisor()
    app = build_graph()
    runs, checks = [], []

    for topic in TEST_TOPICS:
        result = run_team(topic, app)
        log = " ".join(result["team_log"])
        runs.append({"topic": topic, "team_log": result["team_log"], "notes": result["notes"],
                     "analysis": result["analysis"], "review": result["review"],
                     "revisions": result["revision"], "final_report": result["final_report"]})
        short = topic[:40] + "..."
        checks += [
            {"check": f"{short} every agent took part",
             "ok": all(name in log for name in ["Researcher", "Analyst", "Writer", "Reviewer"])},
            {"check": f"{short} the Researcher used tools", "ok": "using 0 searches" not in log},
            {"check": f"{short} final report has a Sources section", "ok": "source" in result["final_report"].lower()},
            {"check": f"{short} revisions stayed within the limit", "ok": result["revision"] <= MAX_REVISIONS + 1},
        ]

    print("\n=== Checks ===")
    all_checks = [{"check": f"Supervisor: {c['case']}", "ok": c["ok"]} for c in supervisor_checks] + checks
    for c in all_checks:
        print(f"  {'PASS' if c['ok'] else 'CHECK'}  {c['check']}")
    print(f"\n{sum(c['ok'] for c in all_checks)}/{len(all_checks)} checks passed.")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"model": MODEL, "max_revisions": MAX_REVISIONS, "runs": runs, "checks": all_checks},
                  f, ensure_ascii=False, indent=2)
    with open("research_report.md", "w", encoding="utf-8") as f:
        f.write(f"# {TEST_TOPICS[0]}\n\n{runs[0]['final_report']}\n")
    print(f"Saved {output_file} and research_report.md. Submit both files with your work.")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lesson 9: a multi-agent research team")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--graph", action="store_true", help="print the graph as a Mermaid diagram (no API key needed)")
    group.add_argument("--check-supervisor", action="store_true", help="test the supervisor's rules (no API key needed)")
    group.add_argument("--topic", metavar="QUESTION", help="research one question")
    group.add_argument("--test", action="store_true", help="run the full test")
    args = parser.parse_args()

    if args.graph:
        print(build_graph().get_graph().draw_mermaid())
        print("Paste this into https://mermaid.live to see the diagram.")
    elif args.check_supervisor:
        check_supervisor()
    elif args.topic:
        run_team(args.topic)
    else:
        run_tests()
