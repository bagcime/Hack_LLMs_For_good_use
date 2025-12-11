# Robust Clinical Lab Extraction Under Adversarial Instructions

This repo contains a small but complete evaluation harness for testing how robust a clinical information–extraction model is to **adversarially modified instructions**.

Given a set of lab-report snippets and ground-truth numeric values (e.g., FEV1, FVC, FENO), the script:

1. Runs a base instruction (clean prompt) for each example.
2. Runs an **updated** instruction where the same task description is suffixed with adversarial / garbage text.
3. Compares the model’s numeric predictions against ground truth.
4. Reports **per-dataset** and **overall** accuracy / error, plus example-level comparisons.

The code is written to be data-agnostic: the CSVs used in my experiments are **not** included in the repo, but you can plug in your own.

---

## Motivation

Large language models are often used to extract structured values from messy clinical text (for example, “return the FEV1% as JSON”).  
However, work on adversarial prompt attacks has shown that small changes to the instruction text can dramatically change model behavior.

This repo is my minimal, end-to-end setup to:

- take *realistic* clinical extraction prompts,
- perturb the instruction description with adversarial suffixes,
- and quantify how much the numeric extraction accuracy degrades.

The idea of adversarial prompt suffixes is inspired by the attacks in the
[llm-attacks](https://github.com/llm-attacks/) project.

---

## High-level design

### Model

- Hugging Face model: `meta-llama/Llama-3.1-8B-Instruct`
- Loaded with `transformers.AutoTokenizer` and `AutoModelForCausalLM`
- Inference on GPU if available, else CPU
- Chat-style prompting with a **system** message and a **user** message

The system prompt is fixed:

> “You are a precise clinical information extraction assistant. You must follow the JSON instructions and extract the requested lab values.”

For each example, generation is constrained by prefixing the output with:

```json
{ "<var_name>": 
