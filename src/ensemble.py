import os
import json
import numpy as np

# --- Configuration & Paths ---
MODEL_A_PREDS = "outputs/model_a/predictions_model_a.jsonl"
MODEL_B_PREDS = "outputs/model_b/predictions_model_b.jsonl"
FINAL_SUBMISSION = "FINAL_SUBMISSION.jsonl"

def load_results(path):
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            data[str(obj['id'])] = float(obj['prediction'])
    return data

def main():
    print("🚀 Initiating Final Ensemble Strategy (60% Model A / 40% Model B)...")
    
    if not os.path.exists(MODEL_A_PREDS) or not os.path.exists(MODEL_B_PREDS):
        print("❌ Error: Missing prediction files. Please run both model_a_llama.py and model_b_deberta.py first!")
        return

    preds_a = load_results(MODEL_A_PREDS)
    preds_b = load_results(MODEL_B_PREDS)

    final_output = []

    for idx in preds_a.keys():
        if idx in preds_b:
            combined_val = (preds_a[idx] * 0.6) + (preds_b[idx] * 0.4)
            final_val = np.clip(combined_val, 1.0, 5.0)

            final_output.append({
                "id": str(idx),
                "prediction": float(final_val)
            })
        else:
            print(f"⚠️ Warning: ID {idx} found in Model A but missing in Model B!")

    with open(FINAL_SUBMISSION, 'w', encoding='utf-8') as f:
        for entry in final_output:
            f.write(json.dumps(entry) + '\n')

    print(f"🏆 Done! Final submission file created at: {FINAL_SUBMISSION}")
    print(f"✅ Total test samples successfully merged: {len(final_output)}")

if __name__ == "__main__":
    main()
