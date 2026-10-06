# Activity 15 — Deploy Your Agent

## Overview

In this activity, you will deploy the **Tools & Function Calling Agent** that you built earlier.

You will learn how to:

* Prepare an agent application for deployment
* Separate the frontend and backend
* Deploy a FastAPI backend
* Connect a Streamlit frontend to the deployed API
* Manage API keys using environment variables
* Test a deployed AI agent

---

## What You Are Deploying

You will deploy the agent you built in the **Tools & Function Calling** activity.

The agent can use:

* Calculator
* Web Search
* Weather

The deployment architecture is:

User
  ↓
Streamlit Frontend
  ↓
FastAPI Backend
  ↓
Tool-Using Agent
  ↓
Gemini
  ↓
Tools


# Part 1 — Prepare the Project

Your deployment folder should contain:


Activity-15-Deploy-Agent/
│
├── agent.py
├── main.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md


### Important Files

agent.py

Contains the tool-using AI agent.

main.p

Contains the FastAPI backend and `/chat` API endpoint.

app.py

Contains the Streamlit frontend.

requirements.txt

Contains the Python packages required by the application.

.gitignore

Prevents sensitive files such as `.env` from being uploaded to GitHub.



# Part 2 — Environment Variables

The Gemini API key should **never be written directly inside your Python code or uploaded to GitHub**.

The application reads the key from an environment variable:


GEMINI_API_KEY


For deployment, add this variable to your hosting platform's environment variables/settings.

Example:


GEMINI_API_KEY = your-api-key


### Security Rule

Never:

* Paste your API key into GitHub
* Share your API key with classmates
* Put your API key directly inside agent.py
* Upload a .env file containing your API key


# Part 3 — Push the Project to GitHub

Upload the deployment files to your GitHub repository.

The activity should be located inside:


Week-05/
└── Activity-15-Deploy-Agent/


### Screenshot 1 — GitHub Project

<img width="1362" height="404" alt="image" src="https://github.com/user-attachments/assets/b823f258-fb27-41f2-9607-379605874716" />


---

# Part 4 — Deploy the FastAPI Backend

The FastAPI application is defined in main.py.

The API provides two endpoints:

### GET `/`

Checks whether the API is running.

### POST `/chat`

Sends a message to the AI agent.

Example request:

json
{
  "message": "What is 25 * 18?"
}


## Deploy with Railway

For this activity, the FastAPI backend is deployed using **Railway**.

After deployment, Railway provides a public URL for your API.

Example:


https://your-project.up.railway.app


Open the URL in your browser.

You should see a message similar to:

`json
{
  "message": "Tools & Function Calling Agent API is running"
}


This confirms that your backend is online.

### Screenshot 2 — Railway Deployment

Add a screenshot showing your Railway service with the deployment status as **Online**.

**[Insert Screenshot 2 — Railway Backend Online Here]**

<img width="1366" height="609" alt="image" src="https://github.com/user-attachments/assets/ebb680c8-f60e-4109-9ac9-ea022f6d27de" />


# Part 5 — Test the FastAPI API

FastAPI automatically provides interactive API documentation through Swagger UI.

Open:


https://your-project.up.railway.app/docs


You should see the available API endpoints.

Open:

**POST /chat

Click:

**Try it out**

Enter:

json
{
  "message": "What is 50 * 98?"
}


Click:

**Execute**

A successful response should contain the calculator result.

Example:

json
{
  "response": {
    "tool_used": "calculator",
    "answer": "25 * 18 is 450."
  }
}


### Screenshot 3 — API Test

Add a screenshot showing the /chat request and successful response.

<img width="1319" height="566" alt="image" src="https://github.com/user-attachments/assets/48879e9a-08d1-4e82-82e6-7bdece648227" />

<img width="1339" height="584" alt="image" src="https://github.com/user-attachments/assets/13ce6140-88bb-4475-a5f3-96f2fdaba6e4" />

# Part 6 — Streamlit Frontend

The Streamlit application is defined in:

text
app.py


The frontend sends the user's message to the deployed FastAPI backend.

The API URL is defined in the Streamlit application:

python
API_URL = "https://your-project.up.railway.app/chat"


The user interacts with Streamlit, while the actual agent runs through the FastAPI backend.

---

# Part 7 — Deploy Streamlit

Deploy the app.py application using **Streamlit Community Cloud**.

When creating the Streamlit application, select your GitHub repository and use:

text
Branch:
main


For the main file path, use:

Week-05/Activity-15-Deploy-Agent/app.py


After deployment, Streamlit provides a public application URL.

Example:


https://your-agent.streamlit.app


---

# Part 8 — Test the Deployed Agent

Open your Streamlit application.

Enter a message such as:

What is 25 * 18?


The request will travel through the deployed system:

text
User
  ↓
Streamlit
  ↓
Railway FastAPI
  ↓
AI Agent
  ↓
Gemini
  ↓
Calculator Tool


You can also test:


Search the web for information about LangGraph.


or:


What is the current weather in Islamabad?


### Screenshot 4 — Deployed Streamlit App

<img width="1353" height="553" alt="image" src="https://github.com/user-attachments/assets/4aae4f85-f6fd-4d00-b926-7b664121489b" />




---

# Deployment Architecture

Your final deployed application looks like this:


                   User
                    │
                    ▼
             Streamlit App
                    │
                    ▼
             Railway / FastAPI
                    │
                    ▼
            Tool-Using Agent
                    │
                    ▼
                 Gemini
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
      Calculator  Web Search  Weather


# Testing Checklist

Before completing the activity, make sure:

* [ ] Project files are uploaded to GitHub
* [ ] API key is stored as an environment variable
* [ ] Railway deployment is Online
* [ ] FastAPI `/docs` opens successfully
* [ ] `/chat` returns a successful response
* [ ] Streamlit application opens successfully
* [ ] Streamlit can communicate with the FastAPI API
* [ ] At least one agent tool works after deployment

---

# Key Takeaways

In this activity, you learned that deploying an AI agent involves more than just running the Python code locally.

A production-style setup separates the application into different components:


Frontend
   ↓
API
   ↓
Agent
   ↓
Model
   ↓
Tools


You also learned the importance of:

* Environment variables
* API endpoints
* Cloud deployment
* API testing
* Error handling
* Keeping API keys secure

---

# Final Result

You have successfully deployed an AI agent that can be accessed through the internet.

Your final application consists of:

**GitHub** → Project source code

**Railway** → FastAPI backend

**Streamlit** → User interface

**Gemini** → AI model

**Tools** → Calculator, Web Search, and Weather
