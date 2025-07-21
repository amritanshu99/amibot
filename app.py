# ✅ app.py

from flask import Flask, request, jsonify
import torch
import pickle
from sentence_transformers import util
from rapidfuzz import fuzz
import re

# 🔹 Load precomputed data
embeddings = torch.load("amibot_data/field_embeddings.pt")

with open("amibot_data/query_list.pkl", "rb") as f:
    query_list = pickle.load(f)

with open("amibot_data/variant_to_response.pkl", "rb") as f:
    variant_to_response = pickle.load(f)

# 🔹 Load model just for encoding user input
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Create Flask app
app = Flask(__name__)

# ✅ Clean input
def clean(text):
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text

@app.route("/amibot", methods=["POST"])
def amibot():
    data = request.get_json()
    user_input = data.get("message", "")

    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    cleaned_input = clean(user_input)
    input_embedding = model.encode(cleaned_input, convert_to_tensor=True)

    # 🔍 Semantic similarity
    cosine_scores = util.cos_sim(input_embedding, embeddings)[0]
    top_idx = torch.argmax(cosine_scores).item()
    top_score = cosine_scores[top_idx].item()

    best_query = query_list[top_idx]
    best_response = variant_to_response[best_query]

    # 🧠 Extra filter: If semantic score is low, check fuzz
    fuzz_score = fuzz.ratio(cleaned_input, best_query)

    if top_score > 0.55 or fuzz_score > 70:
        return jsonify({
            "query_matched": best_query,
            "similarity": round(top_score, 3),
            "fuzz_score": fuzz_score,
            "response": best_response
        })
    else:
        return jsonify({
            "response": "I’m not sure how to answer that. Could you rephrase or ask something else?"
        })

# 🔁 Ping route for cron job or uptime monitor
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok", "message": "AmiBot is alive"}), 200

# ✅ Run
if __name__ == "__main__":
    app.run(debug=True)
