from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import json
import os

app = Flask(__name__)

# ── Model Registry ──
MODEL_DIR = "models"
MODEL_INFO_PATH = os.path.join(MODEL_DIR, "model_info.json")
pipelines = {}       # key -> loaded pipeline
model_info = {}      # full model_info.json contents
default_model = ""   # key of the best model


def load_all_models():
    global pipelines, model_info, default_model

    if not os.path.exists(MODEL_INFO_PATH):
        print(f"Warning: {MODEL_INFO_PATH} not found. Run the notebook first.")
        return

    with open(MODEL_INFO_PATH, "r") as f:
        model_info = json.load(f)

    default_model = model_info.get("best_model", "")

    for key, meta in model_info.get("models", {}).items():
        pkl_path = meta["file"]
        if os.path.exists(pkl_path):
            pipelines[key] = joblib.load(pkl_path)
            print(f"  Loaded: {meta['display_name']} ({pkl_path})")
        else:
            print(f"  Missing: {pkl_path}")

    print(f"\n{len(pipelines)} model(s) loaded. Default: {default_model}")


load_all_models()


# ── Routes ──
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/models")
def list_models():
    """Return available models and their metrics so the frontend can populate the selector."""
    models_list = []
    for key, meta in model_info.get("models", {}).items():
        models_list.append({
            "key": key,
            "display_name": meta["display_name"],
            "is_default": key == default_model,
            "accuracy": meta.get("accuracy"),
            "precision": meta.get("precision"),
            "recall": meta.get("recall"),
            "f1": meta.get("f1"),
            "roc_auc": meta.get("roc_auc"),
        })
    # Sort so default is first, then by roc_auc descending
    models_list.sort(key=lambda m: (not m["is_default"], -(m["roc_auc"] or 0)))
    return jsonify({"default": default_model, "models": models_list})


@app.route("/predict", methods=["POST"])
def predict():
    if not pipelines:
        return jsonify({"error": "No models loaded. Run the notebook to generate models/ first."}), 503

    try:
        data = request.get_json()
        chosen_key = data.pop("model", default_model) or default_model

        if chosen_key not in pipelines:
            return jsonify({"error": f"Model '{chosen_key}' not found."}), 404

        pipeline = pipelines[chosen_key]

        input_data = pd.DataFrame([{
            "person_age": float(data["person_age"]),
            "person_gender": data["person_gender"],
            "person_education": data["person_education"],
            "person_income": float(data["person_income"]),
            "person_emp_exp": float(data["person_emp_exp"]),
            "person_home_ownership": data["person_home_ownership"],
            "loan_amnt": float(data["loan_amnt"]),
            "loan_intent": data["loan_intent"],
            "loan_int_rate": float(data["loan_int_rate"]),
            "loan_percent_income": float(data["loan_amnt"]) / float(data["person_income"]),
            "cb_person_cred_hist_length": float(data["cb_person_cred_hist_length"]),
            "credit_score": int(data["credit_score"]),
            "previous_loan_defaults_on_file": data["previous_loan_defaults_on_file"],
        }])

        prediction = pipeline.predict(input_data)
        probability = pipeline.predict_proba(input_data)[:, 1]

        meta = model_info["models"].get(chosen_key, {})

        return jsonify({
            "prediction": "Approved" if prediction[0] == 1 else "Rejected",
            "probability": round(float(probability[0]) * 100, 2),
            "model_used": meta.get("display_name", chosen_key),
            "model_key": chosen_key,
            "model_roc_auc": meta.get("roc_auc"),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
