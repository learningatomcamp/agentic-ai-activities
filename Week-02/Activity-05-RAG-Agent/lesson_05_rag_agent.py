"""
Lesson 5: RAG for Agents
Activity 01: Agent Knowledge Base (starter file)

You will build an agent that searches a small knowledge base when it needs to,
then answers using what it found.

Most of the code is written for you. Complete the five TODO sections:

    TODO 1  embed()                       turn text into embedding vectors
    TODO 2  build_knowledge_base()        store the documents in ChromaDB
    TODO 3  knowledge_search()            search ChromaDB and filter weak matches
    TODO 4  knowledge_search_declaration  describe the tool so Gemini knows when to use it
    TODO 5  run_rag_agent()               run the tool and send the result back to Gemini

If you run the file before finishing a TODO, it stops with a message naming that TODO.

How to run (from this folder):

    python lesson_05_rag_agent.py --search "stopping rules"   # test retrieval only (TODOs 1-3, no API key needed)
    python lesson_05_rag_agent.py --ask "What is RAG?"        # ask the agent one question
    python lesson_05_rag_agent.py                             # run all test questions, save rag_test_results.json

Set your API key first (see README.md):
    macOS / Linux:  export GEMINI_API_KEY="your-key"
    Windows:        set GEMINI_API_KEY=your-key
"""

import argparse
import json
import os
import time

import chromadb
from chromadb.config import Settings
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # small, free, runs on a laptop
COLLECTION_NAME = "agentic_ai_kb"
TOP_K = 3                              # how many documents to return per search
MIN_SIMILARITY = 0.35                  # ignore matches weaker than this (0 = unrelated, 1 = identical)
MAX_STEPS = 5                          # safety limit for the agent loop (from Lesson 4)


# ---------------------------------------------------------------------------
# The knowledge base: 8 short documents about Agentic AI
# Each document is small enough to be one "chunk".
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
# Part 1: Embeddings
# ---------------------------------------------------------------------------
_embedder = None


def get_embedder() -> SentenceTransformer:
    """Load the embedding model once (the first run downloads about 90 MB)."""
    global _embedder
    if _embedder is None:
        print(f"Loading embedding model '{EMBEDDING_MODEL}'...")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def embed(texts: list[str]) -> list[list[float]]:
    """Turn a list of texts into a list of embedding vectors."""
    embedder = get_embedder()

    # TODO 1: Use embedder.encode(...) to create embeddings for `texts`.
    #         ChromaDB expects plain Python lists, so convert the result with .tolist().
    #         Hint: this is one line.
    raise NotImplementedError("TODO 1: complete embed()")


# ---------------------------------------------------------------------------
# Part 2: Store the documents in ChromaDB
# ---------------------------------------------------------------------------
collection = None   # set by build_knowledge_base()


def build_knowledge_base():
    """Create an in-memory ChromaDB collection and add every document to it."""
    global collection

    chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))

    # Start fresh every run, so re-running never adds duplicates.
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    # "cosine" tells ChromaDB to compare embeddings by meaning (cosine distance).
    collection = chroma_client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    ids = [doc["id"] for doc in KNOWLEDGE_BASE]
    texts = [doc["text"] for doc in KNOWLEDGE_BASE]
    metadatas = [{"title": doc["title"]} for doc in KNOWLEDGE_BASE]

    # TODO 2a: Create embeddings for `texts` using your embed() function.
    embeddings = None

    # TODO 2b: Add the documents to the collection with:
    #          collection.add(ids=..., documents=..., embeddings=..., metadatas=...)

    if embeddings is None:
        raise NotImplementedError("TODO 2a: create the embeddings in build_knowledge_base() (finish TODO 1 first)")
    if collection.count() == 0:
        raise NotImplementedError("TODO 2b: add the documents to the collection in build_knowledge_base()")

    print(f"Knowledge base ready: {collection.count()} documents stored.")
    return collection


# ---------------------------------------------------------------------------
# Part 3: The retrieval tool
# Like every tool from Week 1, it returns a dictionary and never crashes.
# ---------------------------------------------------------------------------
def knowledge_search(query: str) -> dict:
    """Search the knowledge base and return the most relevant documents."""
    if collection is None:
        return {"error": "The knowledge base hasn't been built yet."}

    try:
        # TODO 3a: Embed the query. embed() takes a list, so pass [query] and keep the first vector.
        query_embedding = None

        # TODO 3b: Search the collection for the TOP_K closest documents:
        #          collection.query(query_embeddings=[query_embedding], n_results=TOP_K)
        raw = None

        if raw is None:
            raise NotImplementedError("TODO 3a/3b: embed the query and search the collection in knowledge_search()")
    except NotImplementedError:
        raise
    except Exception as e:
        return {"error": f"Knowledge search failed: {e}"}

    # ChromaDB returns lists of lists (one inner list per query). We sent one query, so we use [0].
    results = []
    for doc_id, text, metadata, distance in zip(
        raw["ids"][0], raw["documents"][0], raw["metadatas"][0], raw["distances"][0]
    ):
        similarity = round(1 - distance, 3)   # cosine distance -> similarity (higher = more similar)

        # TODO 3c: Skip this document if its similarity is below MIN_SIMILARITY.
        #          Hint: use `continue`.

        results.append({"id": doc_id, "title": metadata["title"], "similarity": similarity, "text": text})

    if not results:
        return {"query": query, "results": [], "note": "No relevant documents found in the knowledge base."}
    return {"query": query, "results": results}


# ---------------------------------------------------------------------------
# Part 4: Describe the tool to Gemini
# Gemini decides WHEN to retrieve using only this description.
# ---------------------------------------------------------------------------
knowledge_search_declaration = {
    "name": "knowledge_search",
    # TODO 4: Replace this with a clear description. Say:
    #         - what the knowledge base contains (the topics of the 8 documents)
    #         - when Gemini SHOULD use the tool
    #         - when it should NOT (greetings, small talk, simple maths)
    "description": "TODO 4",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A short search phrase for what to look up, e.g. 'stopping rules for agent loops'.",
            }
        },
        "required": ["query"],
    },
}

TOOL_DECLARATIONS = [knowledge_search_declaration]
AVAILABLE_TOOLS = {"knowledge_search": knowledge_search}

SYSTEM_INSTRUCTION = """You are a helpful assistant for a beginner course on Agentic AI.
You have one tool, knowledge_search, which searches the course knowledge base.
- For questions about agentic AI concepts, call knowledge_search first and answer using only what it returns.
- You may search more than once if a question has several parts.
- For greetings, small talk or simple arithmetic, answer directly without searching.
- If the search returns no results, or the results don't answer the question, say that the knowledge base
  doesn't cover it. Do not guess.
- When you use the knowledge base, end your answer with the titles of the documents you used, like:
  Sources: <title>, <title>"""


# ---------------------------------------------------------------------------
# Part 5: The agent
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


def generate_with_retry(contents, config, max_retries: int = 4):
    """Call Gemini, retrying when the service is busy (503) or rate-limited (429)."""
    for attempt in range(max_retries):
        try:
            return get_client().models.generate_content(model=MODEL, contents=contents, config=config)
        except Exception as e:
            busy = any(code in str(e) for code in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"))
            if busy and attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  Gemini is busy. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def run_rag_agent(question: str, verbose: bool = True) -> dict:
    """Ask the agent a question. It decides whether to search, then answers."""
    if "TODO" in knowledge_search_declaration["description"]:
        raise NotImplementedError("TODO 4: write the description in knowledge_search_declaration")

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        tools=[types.Tool(function_declarations=TOOL_DECLARATIONS)],
    )
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=question)])]
    searches, sources = [], []

    for step in range(1, MAX_STEPS + 1):
        response = generate_with_retry(contents, config)
        if not response.candidates or response.candidates[0].content is None:
            return _result(question, "The model returned no answer.", searches, sources)

        # Keep the model's turn (including any function calls) in the conversation.
        contents.append(response.candidates[0].content)

        function_calls = response.function_calls or []

        # Stopping rule: no tool calls means the agent has its final answer.
        if not function_calls:
            return _result(question, response.text, searches, sources)

        response_parts = []
        for call in function_calls:
            tool_name = call.name
            tool_args = dict(call.args)
            if verbose:
                print(f"  [step {step}] Agent calls {tool_name}({tool_args})")

            # TODO 5a: Run the tool.
            #          Look up tool_name in AVAILABLE_TOOLS and call it with **tool_args.
            #          If tool_name isn't in AVAILABLE_TOOLS, set tool_result to an error dictionary instead.
            tool_result = None

            if tool_result is None:
                raise NotImplementedError("TODO 5a: run the tool in run_rag_agent()")

            # TODO 5b: Wrap the result so Gemini can read it, and add it to response_parts:
            #          types.Part.from_function_response(name=tool_name, response=tool_result)

            # (Provided) Record what was searched and which documents came back.
            if tool_name == "knowledge_search":
                searches.append(tool_args.get("query", ""))
                sources += [r["id"] for r in tool_result.get("results", []) if r["id"] not in sources]
                if verbose:
                    found = [f'{r["title"]} ({r["similarity"]})' for r in tool_result.get("results", [])]
                    print(f"  [step {step}] Found: {found or 'nothing relevant'}")

        if len(response_parts) != len(function_calls):
            raise NotImplementedError("TODO 5b: add each tool result to response_parts in run_rag_agent()")

        # Send the tool results back to Gemini, then loop so it can answer (or search again).
        contents.append(types.Content(role="user", parts=response_parts))

    return _result(question, "Stopped: step limit reached before a final answer.", searches, sources)


def _result(question, answer, searches, sources) -> dict:
    return {"question": question, "used_retrieval": bool(searches), "searches": searches,
            "sources": sources, "answer": answer}


# ---------------------------------------------------------------------------
# Test questions (provided)
# ---------------------------------------------------------------------------
TEST_QUESTIONS = [
    {"question": "What are the three stopping rules for an agent loop?", "should_retrieve": True},
    {"question": "How is RAG different from fine-tuning?", "should_retrieve": True},
    {"question": "Hi! What can you help me with?", "should_retrieve": False},
    {"question": "What is 18 * 24?", "should_retrieve": False},
    {"question": "How do embeddings help an agent remember more than its context window?", "should_retrieve": True},
    {"question": "What does the knowledge base say about evaluating agents with benchmarks?", "should_retrieve": True},
]


def run_tests(output_file: str = "rag_test_results.json"):
    results = []
    for i, test in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[{i}/{len(TEST_QUESTIONS)}] {test['question']}")
        result = run_rag_agent(test["question"])
        result["should_retrieve"] = test["should_retrieve"]
        result["retrieval_decision_ok"] = result["used_retrieval"] == test["should_retrieve"]
        print(f"  Answer: {result['answer']}")
        print(f"  Retrieval decision: {'OK' if result['retrieval_decision_ok'] else 'CHECK THIS'}")
        results.append(result)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"model": MODEL, "min_similarity": MIN_SIMILARITY, "top_k": TOP_K, "results": results},
                  f, ensure_ascii=False, indent=2)

    correct = sum(r["retrieval_decision_ok"] for r in results)
    print(f"\nRetrieval decisions as expected: {correct}/{len(results)}")
    print(f"Saved {output_file}. Submit this file with your work.")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lesson 5: a knowledge-aware RAG agent")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--search", metavar="QUERY", help="test retrieval only (no API key needed)")
    group.add_argument("--ask", metavar="QUESTION", help="ask the agent one question")
    args = parser.parse_args()

    build_knowledge_base()

    if args.search:
        print(json.dumps(knowledge_search(args.search), indent=2))
    elif args.ask:
        result = run_rag_agent(args.ask)
        print(f"\nAnswer: {result['answer']}")
    else:
        run_tests()
