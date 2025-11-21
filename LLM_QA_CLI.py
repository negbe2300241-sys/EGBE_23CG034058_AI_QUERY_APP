import os
import re
import requests
import json
from dotenv import load_dotenv

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash-preview-09-2025"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"

def preprocess_question(question: str) -> str:
    """
    Applies basic text preprocessing as required: lowercasing,
    punctuation removal, and tokenization (then rejoining).

    Args:
        question: The raw natural language question string.

    Returns:
        The preprocessed string of tokens.
    """
    if not question:
        return ""

    # 1. Lowercasing
    text = question.lower()

    # 2. Punctuation removal (keep spaces)
    # This regex keeps only letters, numbers, and spaces
    text = re.sub(r'[^a-z0-9\s]', '', text)

    # 3. Tokenization (splitting by whitespace)
    tokens = text.split()

    # Rejoin tokens into a single string to send to the LLM
    processed_text = " ".join(tokens)
    return processed_text

def get_llm_response(full_prompt: str) -> str:
    """
    Sends the constructed prompt to the Gemini API and returns the response text.
    Implements a basic retry mechanism with exponential backoff for robustness.

    Args:
        full_prompt: The final text prompt to send to the LLM.

    Returns:
        The generated answer text, or an error message.
    """
    if not API_KEY:
        return "ERROR: API Key not found. Please set GEMINI_API_KEY in your .env file."

    headers = {
        "Content-Type": "application/json",
    }

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": full_prompt}
                ]
            }
        ],
        "systemInstruction": {
            # Instruct the model to be concise and directly answer the question
            "parts": [{"text": "You are a concise question-answering system. Answer the user's query directly and accurately, avoiding conversational filler."}]
        }
    }

    # API call with basic error handling
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status() # Raises an HTTPError for bad responses (4xx or 5xx)

        data = response.json()
        
        # Check for generated content
        if data.get('candidates') and data['candidates'][0].get('content'):
            return data['candidates'][0]['content']['parts'][0]['text']
        else:
            # Handle cases where the API returns an error or no text
            error_message = data.get('error', {}).get('message', 'Unknown API Error.')
            return f"API Error: {error_message}"

    except requests.exceptions.HTTPError as e:
        # Specific handling for HTTP errors
        return f"HTTP Error: Could not connect to API. Status code: {e.response.status_code}. Response: {e.response.text}"
    except requests.exceptions.ConnectionError:
        return "Connection Error: Failed to connect to the internet or API endpoint."
    except requests.exceptions.Timeout:
        return "Timeout Error: The request took too long to complete."
    except Exception as e:
        return f"An unexpected error occurred: {e}"

def main():
    """
    Main function for the CLI application loop.
    """
    print("="*60)
    print(f" LLM Q&A CLI Application ({MODEL_NAME})")
    print("="*60)
    print("Enter 'quit' or 'exit' to stop the application.")
    print("-" * 60)

    while True:
        try:
            raw_question = input("\nAsk a question: ")

            if raw_question.lower() in ['quit', 'exit']:
                print("\nExiting Q&A system. Goodbye!")
                break

            if not raw_question.strip():
                continue

            # 1. Apply Preprocessing
            processed_question = preprocess_question(raw_question)
            
            # Display processed question (required by project spec)
            print(f"\n[Processed Query]: {processed_question}")

            # 2. Construct Full Prompt
            # The prompt is constructed to instruct the LLM, using the preprocessed text as the core query.
            full_prompt = f"Using the following preprocessed query, provide a concise answer: '{processed_question}'"

            # 3. Get LLM Response
            print("\n... Thinking ...")
            answer = get_llm_response(full_prompt)

            # 4. Display Final Answer
            print("\n" + "="*60)
            print(" FINAL ANSWER")
            print("="*60)
            print(answer)
            print("-" * 60)

        except KeyboardInterrupt:
            print("\nExiting Q&A system. Goodbye!")
            break
        except Exception as e:
            print(f"An unexpected error occurred in the main loop: {e}")

if __name__ == "__main__":
    main()