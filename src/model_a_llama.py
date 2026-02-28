import os
import json
import torch
import pandas as pd
import numpy as np
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# --- Configuration & Paths ---
DATA_DIR = "data"
OUTPUT_DIR = "outputs/model_a"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, "train.json")
DEV_PATH = os.path.join(DATA_DIR, "dev.json")
TEST_PATH = os.path.join(DATA_DIR, "test.json")
PREDS_PATH = os.path.join(OUTPUT_DIR, "predictions_model_a.jsonl")

MODEL_ID = "NousResearch/Meta-Llama-3-8B"

# --- Data Processing Functions ---
def load_and_fix_data(path):
    try:
        df = pd.read_json(path)
    except ValueError:
        df = pd.read_json(path, orient='index')
        
    if df.shape[1] > 100 and df.shape[0] < df.shape[1]:
        df = df.T
        
    if 'label' not in df.columns:
        df['label'] = df.get('average', df.get('score', 0.0))
    return df

def format_data(row):
    score = row.get('label', 0.0)
    precontext = str(row.get('precontext', '')).strip()
    sentence = str(row.get('sentence', '')).strip()
    ending = row.get('ending', '')
    homonym = row.get('homonym', '')
    definition = row.get('judged_meaning', '')

    has_ending = pd.notna(ending) and str(ending).strip() != ""
    if has_ending:
        instruction = "Task: Evaluate story consistency.\nScale: 1.0 to 5.0."
        context_text = f"Context: {precontext}\nSentence: {sentence}\nEnding: {ending}"
    else:
        instruction = "Task: Evaluate sentence fit.\nScale: 1.0 to 5.0."
        context_text = f"Context: {precontext}\nTarget Sentence: {sentence}"

    text = f"### Instruction:\n{instruction}\n\n### Target: {homonym} ({definition})\n\n### Narrative:\n{context_text}\n\n### Score (1-5):"
    return {"text": text, "label": float(score)}

# --- Main Execution ---
def main():
    print("🔄 Loading Llama-3 Base Model and Tokenizer...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=1,
        quantization_config=bnb_config,
        device_map="auto",
        problem_type="regression"
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["score"], 
        bias="none"
    )
    model = get_peft_model(model, peft_config)
    model.to(torch.float32)

    print("📊 Preparing Dataset...")
    df_train = load_and_fix_data(TRAIN_PATH)
    df_dev = load_and_fix_data(DEV_PATH)
    
    tokenize_func = lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=512)
    
    ds_train = Dataset.from_pandas(df_train.apply(format_data, axis=1).apply(pd.Series)).map(tokenize_func, batched=True)
    ds_dev = Dataset.from_pandas(df_dev.apply(format_data, axis=1).apply(pd.Series)).map(tokenize_func, batched=True)
    combined_ds = concatenate_datasets([ds_train, ds_dev])

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        optim="paged_adamw_8bit",
        fp16=True,
        gradient_checkpointing=True,
        save_strategy="no",
        logging_steps=10,
        report_to="none"
    )

    trainer = Trainer(model=model, args=args, train_dataset=combined_ds)
    
    print("🚀 Starting Training Model A...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("🔮 Running Inference on Test Data...")
    df_test = load_and_fix_data(TEST_PATH)
    
    test_list = []
    for _, row in df_test.iterrows():
        temp_row = row.copy()
        temp_row['label'] = 0.0  # Dummy label for testing
        test_list.append(format_data(temp_row))
        
    test_ds = Dataset.from_pandas(pd.DataFrame(test_list)).map(tokenize_func, batched=True)
    
    predictions = trainer.predict(test_ds)
    test_preds = np.clip(predictions.predictions.flatten(), 1.0, 5.0)

    print(f"📝 Saving predictions to: {PREDS_PATH}")
    with open(PREDS_PATH, 'w', encoding='utf-8') as f:
        for idx, pred in zip(df_test.index.tolist(), test_preds):
            f.write(json.dumps({"id": str(idx), "prediction": float(pred)}) + '\n')

    print("✅ Model A Complete!")

if __name__ == "__main__":
    main()
