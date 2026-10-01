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
        base_url="https://openrouter.ai/api/v1"
    )


def get_model_candidates():
    """Return ordered list of models to try, starting from DEFAULT_MODEL in .env with fallback options."""
    primary = os.getenv("DEFAULT_MODEL", "openrouter/free").strip()
    fallbacks = [
        primary,
        "openrouter/free",
        "nvidia/nemotron-3.5-lightning:free",
        "google/gemma-4-26b-a4b-it:free",
    ]
    # Deduplicate while preserving priority order
    seen = set()
    return [m for m in fallbacks if m and not (m in seen or seen.add(m))]


def generate_completion(client, messages, temperature=0.7, max_tokens=512):
    """Attempt chat completion across available models with automatic fallback on rate-limits (429)."""
    models = get_model_candidates()
    last_error = None
    for model in models:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            content = response.choices[0].message.content
            if content and content.strip():
                return content.strip()
        except Exception as e:
            last_error = e
            print(f"[Model Fallback] Model '{model}' failed: {e}. Trying next available model...")
            continue
    raise last_error or RuntimeError("All available AI models failed to respond. Please try again shortly.")


@app.route("/")
def hello_world():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question")
    if not question:
        return jsonify({"error": "Please provide a question."}), 400
    try:
        client = get_client()
        answer = generate_completion(
            client=client,
            messages=[
                {"role": "system", "content": "Act like a helpful personal assistant"},
                {"role": "user", "content": question}
            ],
            temperature=0.7,
            max_tokens=512
        )
        return jsonify({"response": answer}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/summarize", methods=["POST"])
def summarize():
    email_text = request.form.get("email")
    if not email_text:
        return jsonify({"error": "Please provide text to summarize."}), 400
    try:
        client = get_client()
        summary = generate_completion(
            client=client,
            messages=[
                {"role": "system", "content": "Act like a helpful personal assistant"},
                {"role": "user", "content": f"Summarize this text: {email_text}"}
            ],
            temperature=0.3,
            max_tokens=512
        )
        return jsonify({"response": summary}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)