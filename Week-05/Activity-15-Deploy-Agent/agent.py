
import os
import ast
import operator as op
import time
import requests

from ddgs import DDGS
from google import genai
from google.genai import types


# =========================
# Gemini Setup
# =========================

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

MODEL = "gemini-3.5-flash"


# =========================
# Retry Helper
# =========================

def generate_with_retry(prompt, config=None, max_retries=4):

    for attempt in range(max_retries):

        try:
            return client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=config
            )

        except Exception as e:

            error_text = str(e)

            if "503" in error_text or "UNAVAILABLE" in error_text:

                if attempt < max_retries - 1:

                    wait_time = 2 ** attempt

                    print(
                        f"Gemini temporarily unavailable. "
                        f"Retrying in {wait_time} seconds..."
                    )

                    time.sleep(wait_time)

                else:
                    raise

            else:
                raise


# =========================
# Calculator Tool
# =========================

_ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}


def _safe_eval(node):

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:

        left = _safe_eval(node.left)
        right = _safe_eval(node.right)

        return _ALLOWED_OPERATORS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:

        return _ALLOWED_OPERATORS[type(node.op)](
            _safe_eval(node.operand)
        )

    raise ValueError("Unsupported expression")


def calculator(expression: str):

    try:

        tree = ast.parse(expression, mode="eval")

        result = _safe_eval(tree.body)

        return {
            "expression": expression,
            "result": result
        }

    except Exception:

        return {
            "error": "Invalid or unsupported mathematical expression."
        }


# =========================
# Web Search Tool
# =========================

def web_search(query: str):

    try:

        results = DDGS().text(
            query,
            max_results=3
        )

        simplified = []

        for item in results:

            simplified.append({
                "title": item.get("title"),
                "url": item.get("href"),
                "snippet": item.get("body")
            })

        return simplified

    except Exception as e:

        return {
            "error": f"Search failed: {e}"
        }


# =========================
# Weather Tool
# =========================

def get_weather(city: str):

    try:

        geo_response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={
                "name": city,
                "count": 1,
                "language": "en",
                "format": "json"
            },
            timeout=10
        )

        geo_data = geo_response.json()

        if "results" not in geo_data:

            return {
                "error": f"City '{city}' not found."
            }

        location = geo_data["results"][0]

        latitude = location["latitude"]
        longitude = location["longitude"]

        weather_response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,weather_code",
                "timezone": "auto"
            },
            timeout=10
        )

        weather_data = weather_response.json()

        current = weather_data["current"]

        return {
            "city": city,
            "temperature": current["temperature_2m"],
            "weather_code": current["weather_code"]
        }

    except Exception as e:

        return {
            "error": f"Weather request failed: {e}"
        }


# =========================
# Tool Declarations
# =========================

calculator_declaration = {
    "name": "calculator",
    "description": "Calculates a basic mathematical expression.",
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A mathematical expression such as 25 * 18"
            }
        },
        "required": ["expression"]
    }
}


search_declaration = {
    "name": "web_search",
    "description": "Searches the web for current or specific information.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query."
            }
        },
        "required": ["query"]
    }
}


weather_declaration = {
    "name": "get_weather",
    "description": "Gets the current weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "The name of the city."
            }
        },
        "required": ["city"]
    }
}


TOOL_DECLARATIONS = [
    calculator_declaration,
    search_declaration,
    weather_declaration,
]


AVAILABLE_TOOLS = {
    "calculator": calculator,
    "web_search": web_search,
    "get_weather": get_weather,
}


# =========================
# Tool-Using Agent
# =========================

def run_tool_agent(user_prompt: str):

    config = types.GenerateContentConfig(
        tools=[
            types.Tool(
                function_declarations=TOOL_DECLARATIONS
            )
        ]
    )

    response = generate_with_retry(
        user_prompt,
        config=config
    )

    function_call = None

    for part in response.candidates[0].content.parts:

        if part.function_call:

            function_call = part.function_call
            break

    if function_call is None:

        return response.text

    tool_name = function_call.name
    tool_args = dict(function_call.args)

    print(f"Tool selected: {tool_name}")
    print(f"Arguments: {tool_args}")

    if tool_name not in AVAILABLE_TOOLS:

        return {
            "error": f"Unknown tool: {tool_name}"
        }

    tool_result = AVAILABLE_TOOLS[tool_name](**tool_args)

    print(f"Tool result: {tool_result}")

    final_prompt = f"""
User request:
{user_prompt}

Tool used:
{tool_name}

Tool result:
{tool_result}

Using the tool result, provide a clear and helpful final answer to the user.
"""

    final_response = generate_with_retry(final_prompt)

    return {
        "tool_used": tool_name,
        "answer": final_response.text
    }
