# app.py
"""
Small Flask app for the Mini Tech Challenge.
- Accepts a transcript (web form or JSON POST)
- Calls Groq (requires GROQ_API_KEY) to get summary + sentiment (JSON)
- Appends results to call_analysis.csv with columns: Transcript | Summary | Sentiment | Timestamp
"""

import os
import json
import csv
import datetime
import requests
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

from dotenv import load_dotenv
load_dotenv()


# Config via env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

CSV_FILE = os.getenv("OUTPUT_CSV", "call_analysis.csv")

INDEX_HTML = """
<!doctype html>
<title>Call Analyzer (Groq)</title>
<h2>Mini Tech Challenge — Call Analyzer</h2>
<form method="post" action="/analyze">
  <textarea name="transcript" rows="10" cols="80" placeholder="Paste call transcript here..."></textarea><br/>
  <button type="submit">Analyze</button>
</form>
<hr/>
<p>Or POST JSON to <code>/analyze</code> like <code>{"transcript":"... "}</code></p>
"""

def call_groq_api(transcript: str):
    """
    Call Groq's chat completion endpoint (OpenAI-compatible) and request a JSON output.
    Must return STRICT JSON: {"summary": "...", "sentiment":"Positive|Neutral|Negative"}
    """
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set in environment")

    endpoint = f"{GROQ_BASE}/chat/completions"
    system_prompt = (
        "You are a concise assistant that analyzes customer call transcripts.\n"
        "Return EXACTLY one JSON object and nothing else with two keys:\n"
        '1) "summary" : a 2-3 sentence concise summary of the customer\'s issue.\n'
        '2) "sentiment": one of "Positive", "Neutral", or "Negative".\n'
        "Do not add extra commentary or markdown. Keep summary short and factual."
    )

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript}
        ],
        "temperature": 0
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    resp = requests.post(endpoint, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # Parse response
    content = None
    if isinstance(data, dict):
        if "choices" in data and len(data["choices"]) > 0:
            choice = data["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                content = choice["message"]["content"]
            elif "text" in choice:
                content = choice["text"]

    if content is None:
        content = json.dumps(data)

    # Parse JSON
    parsed = json.loads(content)
    summary = parsed.get("summary", "").strip()
    sentiment = parsed.get("sentiment", "").strip()

    # Normalize sentiment
    sentiment = sentiment.capitalize() if sentiment else "Neutral"
    if sentiment not in ("Positive", "Neutral", "Negative"):
        sentiment = "Neutral"

    return {"summary": summary, "sentiment": sentiment, "raw": content, "api_response": data}

def save_to_csv(transcript, summary, sentiment, csv_file=CSV_FILE):
    header = ["Transcript", "Summary", "Sentiment", "Timestamp"]
    exists = os.path.isfile(csv_file)
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        writer.writerow([transcript, summary, sentiment, datetime.datetime.utcnow().isoformat()])

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)

@app.route("/analyze", methods=["POST"])
def analyze():
    # accept form or JSON
    transcript = None
    if request.form and "transcript" in request.form:
        transcript = request.form["transcript"]
    else:
        body = request.get_json(silent=True) or {}
        transcript = body.get("transcript")

    if not transcript or not transcript.strip():
        return jsonify({"error": "Please provide a 'transcript' field (form or JSON)."}), 400

    try:
        result = call_groq_api(transcript)
        summary = result["summary"]
        sentiment = result["sentiment"]
        save_to_csv(transcript, summary, sentiment)
        print("=== Transcript ===")
        print(transcript)
        print("=== Summary ===")
        print(summary)
        print("=== Sentiment ===")
        print(sentiment)
        return jsonify({"transcript": transcript, "summary": summary, "sentiment": sentiment})
    except requests.HTTPError as e:
        return jsonify({"error": "Upstream API error", "detail": str(e), "response_text": getattr(e.response, "text", "")}), 502
    except Exception as e:
        return jsonify({"error": "Internal server error", "detail": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
