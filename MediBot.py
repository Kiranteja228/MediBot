import os
from dotenv import load_dotenv
import google.generativeai as genai
import streamlit as st
import time
import random
from utils import SAFETY_SETTTINGS
import json
from datetime import datetime
from prompts.conversationPrompt import prompt


# Load environment variables from .env file
load_dotenv()

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def save_chat_history():
    """Save chat history to a JSON file with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_history_{timestamp}.json"
    history = [{
        "role": msg["role"],
        "content": msg["content"],
        "timestamp": msg["timestamp"]
    } for msg in st.session_state.chat_history]
    
    with open(filename, "w") as f:
        json.dump(history, f, indent=2)
    return filename

def add_message(role, content):
    """Add a message to the chat history"""
    st.session_state.chat_history.append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

# Page configuration
st.set_page_config(
    page_title="MediBot",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# MediBot - Your Medical Assistant\nMade by hiliuxg"
    }
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main > div {
        padding: 2rem;
        max-width: 100rem;
    }
    .stChat message {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stTextInput>div>div>input {
        border-radius: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Header section with improved styling
col1, col2 = st.columns([1, 8])
with col1:
    st.image("logo.jpg", width=100)
with col2:
    st.title("MediBot")
    st.caption("Unveiling the Symphony of Healing with Comprehensive Drug Knowledge, Side Effect Insights, and Precision Dosage Guidance.")

# Sidebar configuration
with st.sidebar:
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    with col2:
        if st.button("Save Chat", use_container_width=True):
            filename = save_chat_history()
            st.success(f"Chat saved to {filename}")
    
    st.divider()
    st.markdown("### About")
    st.markdown("MediBot is your AI-powered medical assistant, providing reliable information about medications, side effects, and dosage guidance.")

# Initialize Gemini
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY_CUSTOM"))
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except AttributeError as e:
    st.error("⚠️ Please configure your Gemini API key first.")
    st.stop()

# Model configuration
generation_config = {
    "temperature": 0.5,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
}

try:
    
    model_llm = genai.GenerativeModel('gemini-pro')
    model = genai.GenerativeModel('gemini-pro')
    chat = model.start_chat()
except Exception as e:
    st.error(f"Error initializing model: {str(e)}")
    st.stop()

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input and response
# Chat input and response
if prompt := st.chat_input("Ask me anything about medications...", key="chat_input"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    add_message("user", prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🤔 Thinking...")
        try:
            # Step 1: Process the query with the custom model
            custom_response = ""
            for chunk in model.start_chat().send_message(prompt, stream=True, safety_settings=SAFETY_SETTTINGS):
                custom_response += chunk.text

            # Step 2: Append the medical practitioner chatbot prompt for detailed response
            detailed_prompt = f"Follow these instructions carefully: You are a medical practitioner chatbot providing accurate medical information, adopting a doctor's perspective in your responses. Structure your answers in the following format if applicable:\n{custom_response},"
            
            # Step 3: Use the final prompt as input to the general model for medical advice
            detailed_response = ""
            for chunk in model_llm.start_chat().send_message(detailed_prompt, stream=True, safety_settings=SAFETY_SETTTINGS):
                detailed_response += chunk.text

            # Step 4: Simplify the language and limit the response to concise and easily understandable terms
            simplified_prompt = f"Now, simplify the following response and make it easy to understand for a non-medical person, with a clear and simple explanation:\n{detailed_response}"

            # Step 5: Use the simplified prompt to generate a user-friendly response
            simplified_response = ""
            for chunk in model_llm.start_chat().send_message(simplified_prompt, stream=True, safety_settings=SAFETY_SETTTINGS):
                simplified_response += chunk.text

            # Display the final simplified response
            full_response = simplified_response
            message_placeholder.markdown(full_response)
            add_message("assistant", full_response)

        except genai.types.generation_types.BlockedPromptException as e:
            st.error(f"⚠️ Content blocked: {str(e)}")
        except Exception as e:
            st.error(f"⚠️ An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("*Disclaimer: This is an AI assistant. Always consult healthcare professionals for medical advice.*")