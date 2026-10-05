
# Tools & Function Calling Agent

A simple AI agent built with Gemini that can use tools to complete tasks.

## Tools

- Calculator
- Web Search
- Weather

## Architecture

User
↓
FastAPI
↓
Tool-Using Agent
↓
Gemini
↓
Tools

## API Endpoint

### GET /

Checks whether the API is running.

### POST /chat

Send a user message to the agent.

Example request:

```json
{
  "message": "What is 25 * 18?"
}
