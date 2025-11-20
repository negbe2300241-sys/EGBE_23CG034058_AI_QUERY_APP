import os
import sqlite3
from datetime import datetime, timezone
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import google.generativeai as ai_engine

# 1. Load Environment Config
load_dotenv()

# 2. Configuration
API_SECRET = os.getenv("GEMINI_API_KEY")
DATABASE_FILE = "queries.db"
SERVER_PORT = int(os.getenv("PORT", 5000))

# 3. Initialize Flask (CHANGED 'web_app' to 'app')
app = Flask(__name__, static_folder="static", template_folder="templates")

# 4. Configure AI & DEBUG MODELS
if API_SECRET:
    ai_engine.configure(api_key=API_SECRET)
    
    # --- DEBUGGING: Print available models to terminal on startup ---
    print("\n--- CHECKING AVAILABLE MODELS ---")
    try:
        # This helps you see exactly what models your key allows
        for m in ai_engine.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"Found Model: {m.name}")
        print("---------------------------------\n")
    except Exception as e:
        print(f"Could not list models (Check your API Key): {e}\n")
else:
    print("WARNING: API Key missing.")

def initialize_storage():
    """Creates database if not exists."""
    try:
        db_connection = sqlite3.connect(DATABASE_FILE)
        cursor = db_connection.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT,
                created_at TEXT NOT NULL
            )
        ''')
        db_connection.commit()
        db_connection.close()
        print(f"Database ready: {DATABASE_FILE}")
    except Exception as e:
        print(f"Database Error: {e}")

def record_transaction(user_text, ai_text):
    """Saves to history."""
    try:
        db_connection = sqlite3.connect(DATABASE_FILE)
        cursor = db_connection.cursor()
        timestamp = datetime.now(timezone.utc).isoformat()
        cursor.execute(
            "INSERT INTO queries (question, answer, created_at) VALUES (?, ?, ?)",
            (user_text, ai_text, timestamp)
        )
        db_connection.commit()
        db_connection.close()
    except Exception as e:
        print(f"Save Error: {e}")

@app.route("/") # CHANGED 'web_app' to 'app'
def home_page():
    return render_template("index.html")

@app.route("/api/ask", methods=["POST"]) # CHANGED 'web_app' to 'app'
def process_inquiry():
    data_packet = request.get_json(force=True)
    user_query = data_packet.get("question", "").strip()

    if not user_query:
        return jsonify({"error": "Empty question"}), 400

    if not API_SECRET:
        return jsonify({"answer": "Server Error: API Key missing."})

    ai_response = ""
    
    try:
        # --- FALLBACK LOGIC ---
        # Attempt 1: Try the fast Flash model
        try:
            model = ai_engine.GenerativeModel('models/gemini-2.5-flash')
            result = model.generate_content(user_query)
        except Exception as flash_error:
            print(f"Flash model failed ({flash_error}), switching to Standard Pro...")
            
            # Attempt 2: Fallback to the latest pro model
            model = ai_engine.GenerativeModel('models/gemini-pro-latest')
            result = model.generate_content(user_query)

        if result.parts:
            ai_response = result.text
        else:
            ai_response = "Empty response from AI."

    except Exception as err:
        # If both fail, show the error
        print(f"Final AI Failure: {err}")
        ai_response = f"I encountered an issue: {str(err)}"

    record_transaction(user_query, ai_response)

    return jsonify({
        "question": user_query,
        "answer": ai_response
    })

if __name__ == "__main__":
    initialize_storage()
    app.run(host="0.0.0.0", port=SERVER_PORT, debug=True) # CHANGED 'web_app' to 'app'