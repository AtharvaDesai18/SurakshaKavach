🛡️ Suraksha Kavach: AI-Powered Cybersecurity Toolkit
Suraksha Kavach is a comprehensive, full-stack web application designed to be a one-stop solution for everyday cybersecurity needs. It integrates multiple tools, ranging from a real-time Intrusion Detection System (IDS) to an AI-powered research assistant, into a single, user-friendly dashboard.

✨ Features
Real-time IDS Alert Dashboard: A live monitor that visualizes network anomalies detected by a machine learning model.

Threat Intelligence Lookup: An interactive tool to check the reputation of any IP address or domain against the AbuseIPDB threat database.

Password Health Suite: A client-side suite of tools including a real-time strength checker and a secure password generator.

Phishing Email Analyzer: A tool that analyzes the content of emails to calculate a phishing risk score based on suspicious keywords and links.

AI Cyber Agent: A multimodal AI chat agent powered by gpt-4o-mini that can answer complex cybersecurity questions using both text and image inputs.

🛠️ Technology Stack & Skills
Languages:
Python, JavaScript, HTML, CSS, SQL

AI & Machine Learning:
Frameworks & Libraries: LangChain, scikit-learn, pandas

Models & APIs: OpenAI API (gpt-4o-mini), Isolation Forest

Concepts: AI Agents, Multimodal AI (Text & Image), Unsupervised Learning, Prompt Engineering

Backend & Databases:
Framework: Flask

Database: SQLite

Technologies: REST APIs, Gunicorn

Frontend:
Core: HTML, CSS, JavaScript

Libraries: Chart.js

Templating: Jinja2

DevOps & Tools:
Containerization: Docker

Cloud Deployment: Render

Version Control: Git

🚀 Setup & Installation
Clone the repository:

git clone https://github.com/your-username/SurakshaKavach.git
cd SurakshaKavach

Create and activate a virtual environment:

# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python -m venv venv
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Set up environment variables:

Create a .env file in the root directory.

Add your API keys to the .env file:

OPENAI_API_KEY="your_openai_api_key_here"
ABUSEIPDB_API_KEY="your_abuseipdb_api_key_here"

Usage
The application includes multiple modules. The core Real-time IDS requires a separate script to simulate the data stream.

1. Train the IDS Model (Run once)
.\venv\Scripts\python.exe models/train_model.py

2. Start the Main Application
This command starts the Flask server for the main website.

.\venv\Scripts\python.exe app.py

3. Start the IDS Data Stream (Optional)
To see the live dashboard in action, open a new terminal and run this command.

.\venv\Scripts\python.exe stream/simulate_stream.py

Project Theory & Concepts
Problem Statement
Computer networks are constantly under threat from malicious attacks. Manually monitoring all network activity is impossible due to the sheer volume of data. This project provides an automated solution to learn normal network behavior and flag deviations in real-time.

How This Project Protects a System
This project acts as a foundational layer of security that protects the server and network infrastructure. While it cannot analyze website-specific attacks like SQL injection, it is designed to detect and alert on broader network-level threats such as:

DDoS Attacks: By identifying abnormal flooding patterns in network traffic.

Port Scanning: By detecting systematic attempts to find open and vulnerable ports on a server.
