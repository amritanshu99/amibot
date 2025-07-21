from flask import Flask, request, jsonify
import pickle
import os
from rapidfuzz import fuzz
import re

# 🔧 Load preprocessed data (no transformers)
DATA_DIR = "amibot_data"

with open(os.path.join(DATA_DIR, "field_variants.pkl"), "rb") as f:
    field_variants = pickle.load(f)

with open(os.path.join(DATA_DIR, "field_map.pkl"), "rb") as f:
    field_map = pickle.load(f)

# 🔧 Text cleaning
def clean_text(text):
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text

# 🤖 Fuzzy-only matching response
def get_response(user_input, field_variants, field_map, fuzz_threshold=60):
    original_input = user_input.strip()
    cleaned_input = clean_text(original_input)

    best_field = None
    best_score = -1

    for variant in field_variants:
        score = fuzz.token_set_ratio(cleaned_input, variant.lower())
        if score > best_score:
            best_score = score
            best_field = variant

    if best_score >= fuzz_threshold:
        return {
            "matched": best_field,
            "fuzzy_score": best_score,
            "response": field_map[best_field]
        }
    else:
        return {
            "matched": best_field,
            "fuzzy_score": best_score,
            "response": f"🤖 Sorry, I’m not sure what you meant.\n💡 Did you mean: '{best_field}'?\nPlease rephrase your question."
        }

# 🚀 Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "🧠 AmiBot is running!"

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    user_input = data.get("query", "")

    if not user_input.strip():
        return jsonify({"error": "Empty query provided."}), 400

    result = get_response(user_input, field_variants, field_map)
    return jsonify(result)

# ✅ Cron job ping route
@app.route("/ping")
def ping():
    return "pong"

if __name__ == "__main__":
    app.run(debug=True)
