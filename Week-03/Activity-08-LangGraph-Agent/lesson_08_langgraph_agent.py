"""
Lesson 8: LangChain & LangGraph
Activity 03: LangGraph Agent (starter file)

You will build a Course Helpdesk agent as a LangGraph workflow. It reads each
request, decides what kind it is, and sends it down the right path:

    question     -> retrieve from the knowledge base -> answer with sources
    calculation  -> calculator agent <-> calculator tool (loops until done)
    action       -> draft a message -> ask a human to approve -> send or cancel
    chat         -> reply directly

Conversations are saved with a checkpointer, so the agent remembers earlier
messages in the same thread.

Most of the code is written for you. Complete the seven TODO sections:

    TODO 1  HelpdeskState        define the state the graph passes between nodes
    TODO 2  classify()           node: decide the route for the request
    TODO 3  retrieve()           node: search the knowledge base
    TODO 4  human_approval()     node: pause and ask a human before sending
    TODO 5  route_request()      routing function for the conditional edge
    TODO 6  build_graph()        add the nodes and edges
    TODO 7  build_graph()        compile with a checkpointer

If you run the file before finishing a TODO, it stops with a message naming that TODO.

How to run (from this folder):

    python lesson_08_langgraph_agent.py --graph    # print the graph diagram (no API key needed)
    python lesson_08_langgraph_agent.py --chat     # chat with the agent
    python lesson_08_langgraph_agent.py --test     # run the test requests, save langgraph_test_results.json

Set your API key first (see README.md):
    macOS / Linux:  export GEMINI_API_KEY="your-key"
    Windows:        set GEMINI_API_KEY=your-key
"""

import argparse
import ast
import json
import operator
import os
import uuid
from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
EMBEDDING_MODEL = "gemini-embedding-001"
TOP_K = 3   # documents returned by the retriever


# ---------------------------------------------------------------------------
# The knowledge base: the same 8 documents as Lesson 5
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
# LangChain components (provided): model, retriever, tool
# They're created only when first needed, so --graph works without an API key.
# ---------------------------------------------------------------------------
_llm = None
_retriever = None


def get_llm() -> ChatGoogleGenerativeAI:
    """LangChain MODEL: Gemini wrapped in LangChain's standard chat-model interface."""
    global _llm
    if _llm is None:
        if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
            raise SystemExit("GEMINI_API_KEY is not set. See 'API key setup' in README.md.")
        _llm = ChatGoogleGenerativeAI(model=MODEL)
    return _llm


def get_retriever():
    """LangChain RETRIEVER: embeds the knowledge base once, then finds the closest documents."""
    global _retriever
    if _retriever is None:
        documents = [
            Document(page_content=doc["text"], metadata={"id": doc["id"], "title": doc["title"]})
            for doc in KNOWLEDGE_BASE
        ]
        store = InMemoryVectorStore.from_documents(documents, GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL))
        _retriever = store.as_retriever(search_kwargs={"k": TOP_K})
    return _retriever


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Only numbers and + - * / ** are allowed")


@tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression exactly, for example '0.15 * 2480 + 99'. Use it for any maths."""
    try:
        return str(_safe_eval(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"Error: {e}"


TOOLS = [calculate]

SENT_MESSAGES = []   # a pretend outbox, so nothing is really sent


def send_message(to: str, text: str) -> str:
    """Pretend to send a message. In a real product this would call an email or chat API."""
    SENT_MESSAGES.append({"to": to, "text": text})
    return f"Message sent to {to}."


# ---------------------------------------------------------------------------
# Prompts and structured outputs (provided)
# ---------------------------------------------------------------------------
class RouteDecision(BaseModel):
    """The router's decision."""
    route: Literal["question", "calculation", "action", "chat"] = Field(
        description="Which path should handle the user's latest message")
    reason: str = Field(description="One short sentence explaining the choice")


class MessageDraft(BaseModel):
    """A message the agent wants to send."""
    to: str = Field(description="Who the message is for, e.g. 'my study group' or 'the instructor'")
    text: str = Field(description="The message to send, written politely and briefly")


ROUTER_PROMPT = """You route requests for a course helpdesk about Agentic AI. Pick one route:
- question: asks about agentic AI concepts (agents, tools, the ReAct loop, memory, RAG, embeddings,
  multi-agent systems, agent safety)
- calculation: needs arithmetic or any numeric calculation
- action: asks you to send, email, message or remind someone
- chat: greetings, thanks, small talk, or a follow-up about the conversation so far"""

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful course assistant. Answer the user's latest question using only the context below. "
     "If the context doesn't answer it, say the knowledge base doesn't cover it. Keep answers short and clear."
     "\n\nContext:\n{context}"),
    MessagesPlaceholder("messages"),
])

CALCULATOR_PROMPT = ("You are a careful maths assistant. Use the calculate tool for every calculation, "
                     "never mental maths. Then give the final answer in one short sentence.")

CHAT_PROMPT = ("You are a friendly course helpdesk assistant for a beginner Agentic AI course. "
               "You can answer questions about the course topics, do calculations, and send messages "
               "after the user approves them. Keep replies short.")

DRAFT_PROMPT = "Turn the user's request into a short message to send. Don't add details they didn't give."


# ---------------------------------------------------------------------------
# Small helpers (provided)
# ---------------------------------------------------------------------------
def last_user_message(state) -> str:
    """The text of the most recent message from the user."""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return message.text
    return ""


def conversation(state) -> list:
    """The conversation so far as plain user/assistant text messages (no tool calls)."""
    return [m for m in state["messages"]
            if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and not m.tool_calls and m.text)]


def current_turn(state) -> list:
    """Everything from the latest user message onward, including tool calls and results."""
    messages = state["messages"]
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i:]
    return messages


# ---------------------------------------------------------------------------
# STATE
# The state is the shared "notebook" every node reads from and writes to.
# ---------------------------------------------------------------------------
class HelpdeskState(TypedDict):
    # `add_messages` means: when a node returns messages, ADD them to the list (don't replace it).
    messages: Annotated[list, add_messages]

    # TODO 1: Add the other five fields, each with a type and a short comment:
    #   route          (str)        "question", "calculation", "action" or "chat"
    #   context        (str)        text retrieved from the knowledge base
    #   sources        (list[str])  titles of the retrieved documents
    #   draft          (dict)       the message waiting for approval: {"to": ..., "text": ...}
    #   action_result  (str)        what happened to the message: sent or cancelled


REQUIRED_FIELDS = {"messages", "route", "context", "sources", "draft", "action_result"}


# ---------------------------------------------------------------------------
# NODES
# Each node is a function: it receives the state and returns ONLY the fields it changes.
# ---------------------------------------------------------------------------
def classify(state: HelpdeskState) -> dict:
    """Decide which path should handle the request."""
    router = get_llm().with_structured_output(RouteDecision)

    # TODO 2a: Ask the router to decide. Invoke it with a list of two messages:
    #          SystemMessage(content=ROUTER_PROMPT) and HumanMessage(content=last_user_message(state))
    decision = None

    if decision is None:
        raise NotImplementedError("TODO 2a: invoke the router in classify()")
    print(f"  [classify] route = {decision.route}  ({decision.reason})")

    # TODO 2b: Return a dictionary that updates the "route" field with decision.route.
    raise NotImplementedError("TODO 2b: return the route from classify()")


def retrieve(state: HelpdeskState) -> dict:
    """Search the knowledge base for the user's question."""
    # TODO 3a: Use the retriever to find documents for the user's latest message.
    #          Hint: get_retriever().invoke(...) returns a list of Document objects.
    documents = None

    if documents is None:
        raise NotImplementedError("TODO 3a: search the knowledge base in retrieve()")
    context = "\n\n".join(f"[{d.metadata['title']}]\n{d.page_content}" for d in documents)
    sources = [d.metadata["title"] for d in documents]
    print(f"  [retrieve] found: {sources}")

    # TODO 3b: Return a dictionary that updates the "context" and "sources" fields.
    raise NotImplementedError("TODO 3b: return context and sources from retrieve()")


def answer(state: HelpdeskState) -> dict:
    """Answer the question using the retrieved context (a LangChain chain: prompt | model)."""
    chain = ANSWER_PROMPT | get_llm()
    reply = chain.invoke({"context": state["context"], "messages": conversation(state)})
    text = reply.text + "\n\nSources: " + ", ".join(state["sources"])
    return {"messages": [AIMessage(content=text)]}


def calculator_agent(state: HelpdeskState) -> dict:
    """An LLM with the calculator tool. It may ask for the tool, or give its final answer."""
    model_with_tools = get_llm().bind_tools(TOOLS)
    reply = model_with_tools.invoke([SystemMessage(content=CALCULATOR_PROMPT)] + current_turn(state))
    if reply.tool_calls:
        print(f"  [calculator_agent] calls: {[(c['name'], c['args']) for c in reply.tool_calls]}")
    return {"messages": [reply]}


def chat(state: HelpdeskState) -> dict:
    """Reply directly, using the conversation history."""
    reply = get_llm().invoke([SystemMessage(content=CHAT_PROMPT)] + conversation(state))
    return {"messages": [AIMessage(content=reply.text)]}


def draft_action(state: HelpdeskState) -> dict:
    """Turn the request into a message draft. Nothing is sent yet."""
    drafter = get_llm().with_structured_output(MessageDraft)
    draft = drafter.invoke([SystemMessage(content=DRAFT_PROMPT), HumanMessage(content=last_user_message(state))])
    print(f"  [draft_action] to: {draft.to} | text: {draft.text}")
    return {"draft": draft.model_dump()}


def human_approval(state: HelpdeskState) -> dict:
    """HUMAN-IN-THE-LOOP: pause the graph and wait for a person to approve the draft."""
    draft = state["draft"]

    # TODO 4: Pause the graph and wait for a human decision. Call interrupt() with a dictionary
    #         that shows the person what they're approving:
    #             {"question": "Send this message? (yes/no)", "to": draft["to"], "text": draft["text"]}
    #         interrupt() returns whatever the human answers when the graph is resumed.
    decision = None

    if decision is None:
        raise NotImplementedError("TODO 4: call interrupt() in human_approval()")

    if str(decision).strip().lower() in ("yes", "y", "approve", "approved"):
        result = send_message(draft["to"], draft["text"])
    else:
        result = "Cancelled. Nothing was sent."
    print(f"  [human_approval] {result}")
    return {"action_result": result, "messages": [AIMessage(content=result)]}


# ---------------------------------------------------------------------------
# ROUTING
# A routing function reads the state and returns the NAME of the next node.
# ---------------------------------------------------------------------------
def route_request(state: HelpdeskState) -> str:
    """Pick the next node based on the route chosen by classify()."""
    # TODO 5: Return the name of the next node for state["route"]:
    #           "question"    -> "retrieve"
    #           "calculation" -> "calculator_agent"
    #           "action"      -> "draft_action"
    #           "chat"        -> "chat"
    #         If the route is anything else, return "chat" as a safe default.
    #         Hint: a dictionary and .get(key, default) does this in two lines.
    raise NotImplementedError("TODO 5: complete route_request()")


# ---------------------------------------------------------------------------
# EDGES: build and compile the graph
# ---------------------------------------------------------------------------
def build_graph(checkpointer=None):
    graph = StateGraph(HelpdeskState)

    graph.add_node("classify", classify)
    graph.add_node("retrieve", retrieve)
    graph.add_node("answer", answer)
    graph.add_node("calculator_agent", calculator_agent)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("chat", chat)
    graph.add_node("draft_action", draft_action)
    graph.add_node("human_approval", human_approval)

    missing = REQUIRED_FIELDS - set(HelpdeskState.__annotations__)
    if missing:
        raise NotImplementedError(f"TODO 1: add these fields to HelpdeskState: {sorted(missing)}")

    # Normal edges (provided): the question path.
    graph.add_edge(START, "classify")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)

    # TODO 6a: CONDITIONAL edge from "classify". Use route_request to choose the next node,
    #          and list the four possible destinations:
    #          graph.add_conditional_edges("classify", route_request, ["retrieve", "calculator_agent", "draft_action", "chat"])

    # TODO 6b: The calculator LOOP.
    #          - conditional edge from "calculator_agent" using tools_condition
    #            (it returns "tools" if the model asked for a tool, otherwise END)
    #          - normal edge from "tools" back to "calculator_agent"

    # TODO 6c: The approval path and the chat path.
    #          - "draft_action" -> "human_approval" -> END
    #          - "chat" -> END

    # TODO 7: Compile the graph WITH a checkpointer, so conversations and approval pauses are saved.
    #         Change the next line to:  graph.compile(checkpointer=checkpointer or InMemorySaver())
    app = graph.compile()

    _check_graph(app)
    return app


def _check_graph(app):
    """(Provided) Give a clear message if an edge or the checkpointer is missing."""
    edges = app.get_graph().edges
    sources = {e.source for e in edges}
    targets = {e.target for e in edges}
    for node in ["classify", "calculator_agent", "tools", "draft_action", "human_approval", "chat"]:
        if node not in sources:
            raise NotImplementedError(f"TODO 6: node '{node}' has no outgoing edge in build_graph()")
    for node in ["retrieve", "calculator_agent", "draft_action", "chat", "human_approval"]:
        if node not in targets:
            raise NotImplementedError(f"TODO 6: nothing leads to node '{node}' in build_graph()")
    if app.checkpointer is None:
        raise NotImplementedError("TODO 7: compile the graph with a checkpointer in build_graph()")


# ---------------------------------------------------------------------------
# Running one turn (provided)
# ---------------------------------------------------------------------------
def run_turn(app, message: str, thread_id: str, approve=None) -> dict:
    """Send one message in a conversation thread. Handles the approval pause if the graph stops for it."""
    config = {"configurable": {"thread_id": thread_id}}
    fresh = {"messages": [HumanMessage(content=message)], "route": "", "context": "",
             "sources": [], "draft": {}, "action_result": ""}
    result = app.invoke(fresh, config)

    if "__interrupt__" in result:
        request = result["__interrupt__"][0].value
        print(f"\n  APPROVAL NEEDED -> to: {request['to']}\n                     text: {request['text']}")
        decision = approve if approve is not None else input("  Send this message? (yes/no): ")
        print(f"  Human says: {decision}")
        result = app.invoke(Command(resume=decision), config)

    return result


def summarise_turn(result: dict) -> dict:
    tools_used = [m.name for m in current_turn(result) if isinstance(m, ToolMessage)]
    return {"route": result.get("route", ""), "reply": result["messages"][-1].text,
            "sources": result.get("sources", []), "tools_used": tools_used,
            "action_result": result.get("action_result", "")}


# ---------------------------------------------------------------------------
# Tests (provided)
# ---------------------------------------------------------------------------
TESTS = [
    {"message": "What is prompt injection?", "expect_route": "question",
     "expect_source": "Agent safety and guardrails"},
    {"message": "What is 15% of 2,480, plus 99?", "expect_route": "calculation",
     "expect_tool": "calculate", "expect_in_reply": "471"},
    {"message": "Hi! What can you help me with?", "expect_route": "chat"},
    {"message": "Send a reminder to my study group that we meet at 6pm on Friday.", "expect_route": "action",
     "approve": "yes", "expect_action": "sent"},
    {"message": "Email my instructor that I'll miss tomorrow's class.", "expect_route": "action",
     "approve": "no", "expect_action": "cancelled"},
]


def run_tests(output_file: str = "langgraph_test_results.json"):
    app = build_graph()
    results, checks = [], []

    for i, test in enumerate(TESTS, 1):
        print(f"\n[{i}/{len(TESTS)}] You: {test['message']}")
        turn = summarise_turn(run_turn(app, test["message"], thread_id=f"test-{i}", approve=test.get("approve")))
        print(f"  Agent: {turn['reply']}")
        results.append({"message": test["message"], **turn})

        checks.append({"check": f"'{test['message']}' routed to {test['expect_route']}",
                       "ok": turn["route"] == test["expect_route"]})
        if "expect_source" in test:
            checks.append({"check": f"Retrieved '{test['expect_source']}'", "ok": test["expect_source"] in turn["sources"]})
        if "expect_tool" in test:
            checks.append({"check": f"Used the {test['expect_tool']} tool", "ok": test["expect_tool"] in turn["tools_used"]})
        if "expect_in_reply" in test:
            checks.append({"check": f"Reply contains {test['expect_in_reply']}",
                           "ok": test["expect_in_reply"] in turn["reply"].replace(",", "")})
        if "expect_action" in test:
            checks.append({"check": f"Message was {test['expect_action']} after human said '{test['approve']}'",
                           "ok": test["expect_action"] in turn["action_result"].lower()})

    # Persistence: two turns in the same thread.
    print("\n[Persistence] Two messages in the same thread:")
    thread = "test-memory"
    for message in ["What is RAG?", "Can you explain that in one sentence for a 10-year-old?"]:
        print(f"\n  You: {message}")
        turn = summarise_turn(run_turn(app, message, thread_id=thread))
        print(f"  Agent: {turn['reply']}")
        results.append({"message": message, "thread": thread, **turn})
    saved = app.get_state({"configurable": {"thread_id": thread}}).values["messages"]
    checks.append({"check": "Checkpointer kept both turns in the same thread (4+ messages)", "ok": len(saved) >= 4})

    print("\n=== Checks ===")
    for c in checks:
        print(f"  {'PASS' if c['ok'] else 'CHECK'}  {c['check']}")
    print(f"\n{sum(c['ok'] for c in checks)}/{len(checks)} checks passed.")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"model": MODEL, "results": results, "checks": checks}, f, ensure_ascii=False, indent=2)
    print(f"Saved {output_file}. Submit this file with your work.")


def run_chat():
    app = build_graph()
    thread_id = str(uuid.uuid4())
    print("Chat with the Course Helpdesk. Type 'new' for a new thread, 'quit' to exit.")
    while True:
        message = input("\nYou: ").strip()
        if not message:
            continue
        if message.lower() in ("quit", "exit"):
            break
        if message.lower() == "new":
            thread_id = str(uuid.uuid4())
            print("(New thread started. The agent won't see the previous conversation.)")
            continue
        result = run_turn(app, message, thread_id)
        print(f"Agent: {result['messages'][-1].text}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lesson 8: a LangGraph helpdesk agent")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--graph", action="store_true", help="print the graph as a Mermaid diagram (no API key needed)")
    group.add_argument("--chat", action="store_true", help="chat with the agent")
    group.add_argument("--test", action="store_true", help="run the test requests")
    args = parser.parse_args()

    if args.graph:
        print(build_graph().get_graph().draw_mermaid())
        print("Paste this into https://mermaid.live to see the diagram.")
    elif args.chat:
        run_chat()
    else:
        run_tests()
