import os
import json
import gc
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, set_seed

# --- Configuration & Paths ---
DATA_DIR = "data"
OUTPUT_DIR = "outputs/model_b"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, "train.json")
DEV_PATH = os.path.join(DATA_DIR, "dev.json")
TEST_PATH = os.path.join(DATA_DIR, "test.json")
PREDS_PATH = os.path.join(OUTPUT_DIR, "predictions_model_b.jsonl")

MODELS = {
    "standard": "microsoft/deberta-v3-large",
    "nli": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
}

MIN_SCORE, MAX_SCORE = 1.0, 5.0
set_seed(42)

# --- Data Loading & Datasets ---
def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, dict): 
            return list(data.values())
        return data

class StandardDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_len=512):
        self.data = data_list
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self): 
        return len(self.data)
        
    def __getitem__(self, index):
        item = self.data[index]
        homonym = item.get('homonym', '')
        sentence = item.get('sentence', '')
        
        if homonym and homonym in sentence:
            highlighted = sentence.replace(homonym, f'" {homonym} "')
        else: 
            highlighted = sentence
            
        full_story = f"{item.get('precontext', '')} {highlighted} {item.get('ending', '')}".strip()
        
        val = float(item.get('average', 0.0))
        normalized = (val - MIN_SCORE) / (MAX_SCORE - MIN_SCORE) if val > 0 else 0.0

        inputs = self.tokenizer(
            item.get('judged_meaning', ''), 
            full_story, 
            max_length=self.max_len, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].flatten(), 
            'attention_mask': inputs['attention_mask'].flatten(), 
            'labels': torch.tensor(float(normalized), dtype=torch.float)
        }

class NLIDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_len=512):
        self.data = data_list
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self): 
        return len(self.data)
        
    def __getitem__(self, index):
        item = self.data[index]
        homonym = item.get('homonym', '')
        sentence = item.get('sentence', '')
        
        if homonym and homonym in sentence:
            highlighted = sentence.replace(homonym, f'" {homonym} "')
        else: 
            highlighted = sentence
            
        premise = f"{item.get('precontext', '')} {highlighted} {item.get('ending', '')}".strip()
        hypothesis = f'The word "{homonym}" implies: {item.get("judged_meaning", "")}'
        
        val = float(item.get('average', 0.0))
        normalized = (val - MIN_SCORE) / (MAX_SCORE - MIN_SCORE) if val > 0 else 0.0

        inputs = self.tokenizer(
            premise, 
            hypothesis, 
            max_length=self.max_len, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].flatten(), 
            'attention_mask': inputs['attention_mask'].flatten(), 
            'labels': torch.tensor(float(normalized), dtype=torch.float)
        }

# --- Training & Inference Logic ---
def run_test_cycle(model_type, model_name, full_data, test_data_values):
    print(f"\n🚀 STARTING DEBERTA for: {model_type.upper()}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if model_type == "standard":
        full_ds = StandardDataset(full_data, tokenizer)
        test_ds = StandardDataset(test_data_values, tokenizer)
    else:
        full_ds = NLIDataset(full_data, tokenizer)
        test_ds = NLIDataset(test_data_values, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1, problem_type="regression", ignore_mismatched_sizes=True
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, f"tmp_{model_type}"),
        num_train_epochs=5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        weight_decay=0.01,
        save_strategy="no",
        fp16=True,
        report_to="none"
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=full_ds)
    
    print("🔥 Training...")
    trainer.train()

    print("🔮 Predicting on TEST...")
    output = trainer.predict(test_ds)
    logits = output.predictions[0] if isinstance(output.predictions, tuple) else output.predictions
    preds = logits.flatten() * (MAX_SCORE - MIN_SCORE) + MIN_SCORE
    preds = np.clip(preds, MIN_SCORE, MAX_SCORE)

    del model; del trainer; del tokenizer; gc.collect(); torch.cuda.empty_cache()
    return preds

# --- Main Execution ---
def main():
    print("📥 Loading and merging Train + Dev data for Model B...")
    full_data = load_data(TRAIN_PATH) + load_data(DEV_PATH)

    with open(TEST_PATH, 'r', encoding='utf-8') as f:
        raw_test_dict = json.load(f)
    
    test_ids = list(raw_test_dict.keys())
    test_data_values = list(raw_test_dict.values())

    preds_std = run_test_cycle("standard", MODELS["standard"], full_data, test_data_values)
    preds_nli = run_test_cycle("nli", MODELS["nli"], full_data, test_data_values)

    print("\n⚖️ CALCULATING FINAL HYBRID DEBERTA ENSEMBLE...")
    final_ensemble_preds = (preds_std + preds_nli) / 2

    results = [{"id": pid, "prediction": float(p)} for pid, p in zip(test_ids, final_ensemble_preds)]
    
    with open(PREDS_PATH, 'w', encoding='utf-8') as f:
        for entry in results:
            f.write(json.dumps(entry) + '\n')
            
    print(f"✅ Model B predictions saved in: {PREDS_PATH}")

if __name__ == "__main__":
    main()
