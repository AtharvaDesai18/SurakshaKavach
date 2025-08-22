# app.py

from flask import Flask, request, jsonify, render_template, Response
import joblib
import pandas as pd
import numpy as np
import os
import threading
import time
import requests 
import re 
import sqlite3
from dotenv import load_dotenv

# --- LANGCHAIN AGENT IMPORTS ---
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# --- API KEYS ---
ABUSEIPDB_API_KEY = os.getenv('ABUSEIPDB_API_KEY')

# --- DATABASE SETUP ---
DATABASE_FILE = 'database.db'
def init_db():
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            source_ip TEXT,
            destination_ip TEXT,
            destination_port INTEGER,
            label TEXT
        )
    ''')
    conn.commit()
    conn.close()
    print("Database initialized successfully.")


# --- LANGCHAIN AGENT SETUP ---
def setup_agent():
    try:
        if not os.getenv("OPENAI_API_KEY"):
            print("!!! WARNING: OPENAI_API_KEY not found. Cyber Agent will be disabled.")
            return None
            
        llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
                You are 'Suraksha Kavach', a friendly and knowledgeable AI Cyber Agent. 
                Your personality is helpful, reassuring, and clear.

                When a user asks a question:
                1. First, use the tools at your disposal to gather the most accurate and up-to-date information.
                2. Synthesize the information you've found into a comprehensive but easy-to-understand answer.
                3. Structure your final response using clear markdown headings.
                4. **IMPORTANT**: For any numbered lists, you MUST use sequential numbers (1., 2., 3., etc.).
                5. Use relevant emojis to make the information engaging and visually appealing.
            """),
            ("human", "{query}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        tools = [search_tool, wiki_tool]
        agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
        print("LangChain Cyber Agent initialized successfully.")
        return agent_executor
    except Exception as e:
        print(f"!!! Failed to initialize LangChain Agent: {e}")
        return None

agent_executor = setup_agent()


# --- IDS MODEL LOADING ---
model = None
scaler = None
try:
    script_dir = os.path.dirname(__file__)
    MODEL_PATH = os.path.normpath(os.path.join(script_dir, 'models', 'isolation_forest_model.joblib'))
    SCALER_PATH = os.path.normpath(os.path.join(script_dir, 'models', 'scaler.joblib'))
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("IDS Model and scaler loaded successfully.")
except Exception as e:
    print(f"!!! An error occurred while loading IDS model/scaler: {e}")


# --- SETUP & IN-MEMORY STORAGE ---
features = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Fwd Packet Length Max',
    'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Bwd Packet Length Max', 'Bwd Packet Length Min'
]
data_lock = threading.Lock()
stats = {"packets_processed": 0}

# --- PAGE ROUTES ---
@app.route('/')
def ids_dashboard():
    return render_template('ids_dashboard.html')

@app.route('/threat-intel', methods=['GET', 'POST'])
def threat_lookup_page():
    result = None
    error = None
    if request.method == 'POST':
        ip_address = request.form.get('ip_address')
        if not ip_address:
            error = "Please enter an IP address."
        elif not ABUSEIPDB_API_KEY:
            error = "AbuseIPDB API key is not configured. Please add it to your .env file."
        else:
            api_url = 'https://api.abuseipdb.com/api/v2/check'
            headers = {'Accept': 'application/json', 'Key': ABUSEIPDB_API_KEY}
            params = {'ipAddress': ip_address, 'maxAgeInDays': '90'}
            try:
                response = requests.get(api_url, headers=headers, params=params)
                response.raise_for_status()
                api_result = response.json()
                result = api_result.get('data')
                if not result:
                    error = f"No data found for IP address: {ip_address}"
            except requests.exceptions.HTTPError as http_err:
                error = f"API Error: {http_err.response.status_code}. Check your API key and the IP address."
            except Exception as e:
                error = f"An unexpected error occurred: {e}"
    return render_template('threat_lookup.html', result=result, error=error)

@app.route('/password-tools')
def password_tools():
    return render_template('password_tools.html')

@app.route('/phishing-analyzer', methods=['GET', 'POST'])
def phishing_analyzer():
    result = None
    if request.method == 'POST':
        email_content = request.form.get('email_content', '')
        urls_found = re.findall(r'(https?://[^\s]+)', email_content)
        phishing_keywords = [
            'urgent', 'verify', 'account', 'suspended', 'login', 'password',
            'security', 'alert', 'confirm', 'immediately', 'invoice', 'payment'
        ]
        keywords_found = [word for word in phishing_keywords if word in email_content.lower()]
        risk_score = min(25 * (len(urls_found) > 0) + 10 * len(keywords_found), 100)
        result = {
            "risk_score": risk_score,
            "urls_found": urls_found,
            "keywords_found": keywords_found
        }
    return render_template('phishing_analyzer.html', result=result)

@app.route('/cyber-agent', methods=['GET', 'POST'])
def cyber_agent():
    if request.method == 'GET':
        # *** FIX: Pass a variable to the template to hide the footer ***
        return render_template('cyber_agent.html', hide_footer=True)

    if request.method == 'POST':
        if not agent_executor:
            return Response("Agent not initialized. Check server logs and ensure OPENAI_API_KEY is set.", status=500)
            
        query = request.json.get('query')
        if not query:
            return Response("No query provided.", status=400)

        def generate():
            try:
                for chunk in agent_executor.stream({"query": query}):
                    if "output" in chunk:
                        yield chunk["output"]
            except Exception as e:
                print(f"Error during agent execution: {e}")
                yield "Sorry, an error occurred while processing your request."

        return Response(generate(), mimetype='text/plain')


# --- API ENDPOINTS FOR IDS DASHBOARD ---
@app.route('/api/detect', methods=['POST'])
def detect_anomaly():
    if not model or not scaler:
        return jsonify({"status": "error", "message": "Model or scaler is not loaded."}), 500
    
    data = request.get_json()
    if not data: return jsonify({"status": "error", "message": "No data provided"}), 400
    
    with data_lock:
        stats["packets_processed"] += 1

    df_point = pd.DataFrame([data])
    df_features = df_point[features].copy()
    df_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_features.fillna(0, inplace=True)

    X_scaled = scaler.transform(df_features)
    prediction = model.predict(X_scaled)
    
    if prediction[0] == -1:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO alerts (timestamp, source_ip, destination_ip, destination_port, label) VALUES (?, ?, ?, ?, ?)",
            (
                time.strftime('%Y-%m-%d %H:%M:%S'),
                data.get('Source IP'),
                data.get('Destination IP'),
                data.get('Destination Port'),
                data.get('Label')
            )
        )
        conn.commit()
        conn.close()
        return jsonify({"status": "anomaly detected"})
    
    return jsonify({"status": "normal"})

@app.route('/api/data', methods=['GET'])
def get_data():
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT 50")
    alerts_from_db = cursor.fetchall()
    alerts_list = [dict(row) for row in alerts_from_db]
    cursor.execute("SELECT COUNT(*) FROM alerts")
    total_anomalies = cursor.fetchone()[0]
    conn.close()

    with data_lock:
        current_stats = stats.copy()
        current_stats["anomalies_detected"] = total_anomalies
        anomaly_rate = (total_anomalies / current_stats["packets_processed"] * 100) if current_stats["packets_processed"] > 0 else 0
        current_stats["anomaly_rate"] = anomaly_rate
        
    return jsonify({"alerts": alerts_list, "stats": current_stats})

@app.route('/api/clear', methods=['POST'])
def clear_data():
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM alerts")
    conn.commit()
    conn.close()
    
    with data_lock:
        stats["packets_processed"] = 0
        
    return jsonify({"status": "success"})

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)
