# SemEval-2026 Task 5 (AmbiStory): Team JCT 2026

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Rank](https://img.shields.io/badge/SemEval_Rank-22%2F79-success.svg)]()

This repository contains the official code for **Team "JCT 2026"** (Chava Laufer, Batel Sara Turjeman, and Chaya Liebeskind) for [SemEval-2026 Task 5: Rating Plausibility of Word Senses in Ambiguous Sentences (AmbiStory)](https://semeval.github.io/SemEval2026/).

Our system introduces a **Hybrid LLM-NLI Ensemble**, integrating a generative Large Language Model (Llama-3 8B, fine-tuned via LoRA) with a dual-expert bidirectional cross-encoder (DeBERTa-v3-large) optimized for both semantic similarity and Natural Language Inference (NLI).

🏆 **Results:** Our architecture ranked **22nd out of 79 systems**, achieving a Spearman Rank Correlation of **0.71** and a Soft Accuracy of **82.04%**.

## ⚙️ Installation

```bash
git clone [https://github.com/YOUR-USERNAME/semeval2026-task5-ambistory.git](https://github.com/YOUR-USERNAME/semeval2026-task5-ambistory.git)
cd semeval2026-task5-ambistory
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
