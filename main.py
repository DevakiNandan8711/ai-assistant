from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
MY_API_KEY = os.getenv("MY_API_KEY")

app = Flask(__name__)


def get_client():
    """Create the OpenRouter client lazily so Flask can start without a key."""
    if not MY_API_KEY:
        raise ValueError("MY_API_KEY is missing. Add it to your .env file.")
    return OpenAI(
        api_key=MY_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )


@app.route("/")
def hello_world():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question")
    client = get_client()
    try:
        response = client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[
                {"role": "system", "content": "Act like a helpful personal assistant"},
                {"role": "user", "content": question}
            ],
            temperature=0.7,
            max_tokens=512
        )
        answer = response.choices[0].message.content.strip()
        return jsonify({"response": answer}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/summarize", methods=["POST"])
def summarize():
    email_text = request.form.get("email")
    client = get_client()
    try:
        response = client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[
                {"role": "system", "content": "Act like a helpful personal assistant"},
                {"role": "user", "content": f"Summarize this text: {email_text}"}
            ],
            temperature=0.3,
            max_tokens=512
        )
        summary = response.choices[0].message.content.strip()
        return jsonify({"response": summary}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)