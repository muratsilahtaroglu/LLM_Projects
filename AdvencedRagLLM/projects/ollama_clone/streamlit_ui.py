from dotenv import load_dotenv
import os
import streamlit as st
import requests

# Load environment variables from .env file
load_dotenv()
PORT = os.getenv("PORT", "8087")
# Use environment variables in the script
API_URL = os.getenv("API_URL", f"http://127.0.0.1:{PORT}/api/get_clone_response")  # Default value as fallback
APP_TITLE = os.getenv("APP_TITLE", "CloneAI Query Interface")
APP_SUBTITLE = os.getenv("APP_SUBTITLE", "Messaging with the CloneAI model")
PLACEHOLDER_TEXT = os.getenv("PLACEHOLDER_TEXT", "Hello, I'm CloneAI. How can I help you?")
SEND_BUTTON_TEXT = os.getenv("SEND_BUTTON_TEXT", "Send")
SPINNER_TEXT = os.getenv("SPINNER_TEXT", "Please wait, this might take some time. Retrieving response from CloneAI...")
WARNING_TEXT = os.getenv("WARNING_TEXT", "Please enter a valid query.")
ERROR_TEXT_TEMPLATE = os.getenv("ERROR_TEXT_TEMPLATE", "An error occurred: {error}")
RESPONSE_ERROR_TEXT_TEMPLATE = os.getenv("RESPONSE_ERROR_TEXT_TEMPLATE", "Error {status_code}: {error_detail}")


# Streamlit UI
st.title(APP_TITLE)
st.subheader(APP_SUBTITLE)

# Input field for the user query
query = st.text_input(PLACEHOLDER_TEXT)

# Submit button
if st.button(SEND_BUTTON_TEXT):
    if query.strip():
        with st.spinner(SPINNER_TEXT):
            try:
                # Send POST request to the FastAPI endpoint
                response = requests.post(
                    API_URL,
                    json={"query": query}
                )
                if response.status_code == 200:
                    # Parse the response
                    response_data = response.json()
                    
                    # Extract the first response value
                    first_response = response_data.get("response", ["No response received"])[0] if isinstance(response_data.get("response"), list) else "No response received"
                    
                    # Display the first response
                    st.write(f"**Response:** {first_response}")
                else:
                    # Handle API error
                    error_detail = response.json().get("detail", "Unknown error")
                    st.error(RESPONSE_ERROR_TEXT_TEMPLATE.format(status_code=response.status_code, error_detail=error_detail))
            except Exception as e:
                # Handle request exception
                st.error(ERROR_TEXT_TEMPLATE.format(error=str(e)))
    else:
        st.warning(WARNING_TEXT)