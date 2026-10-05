
import streamlit as st
import requests

st.set_page_config(
    page_title="Tools & Function Calling Agent"
)

st.title("Tools & Function Calling Agent")

API_URL = "https://agentic-ai-activities-production.up.railway.app/chat"

user_message = st.text_input("Enter your message:")

if st.button("Send"):

    if user_message:

        response = requests.post(
            API_URL,
            json={"message": user_message}
        )

        if response.status_code == 200:
            data = response.json()
            st.write(data["response"])
        else:
            st.error(f"API Error: {response.status_code}")
