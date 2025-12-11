#!/usr/bin/env python3
"""
Evaluate original vs updated instruction descriptions on ALL lab sub-datasets.

Datasets handled:
  - TestZero_FEF25_75_Liter.csv          -> FEF25_75_Liter
  - TestZero_FEF25_75_Pre_Percentage.csv -> FEF25_75_Pre_Percentage
  - TestZero_FENO.csv                    -> FENO
  - TestZero_FEV1_fvc.csv                -> FEV1_fvc
  - TestZero_fev1_percentage_pre.csv     -> fev1_percentage_pre
  - TestZero_fvc_liter_pre.csv           -> fvc_liter_pre
"""

import json
import math
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ==========================
# 0) GLOBAL CONFIG
# ==========================

HF_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SYSTEM_PROMPT = (
    "You are a precise clinical information extraction assistant. "
    "You must follow the JSON instructions and extract the requested lab values."
)

# Strict numeric equality (0.8 vs 0.80 counts as equal, 0.8 vs 0.81 ≠)
ERROR_TOL = 1e-4
GEN_MAX_NEW_TOKENS = 8
MAX_PRINT_ROWS = 10

torch.set_grad_enabled(False)

# ==========================
# 1) PER-DATASET CONFIG
# ==========================

DATASETS = [
    {
        "name": "FEF25_75_Liter",
        "csv_path": "TestZero_FEF25_75_Liter.csv",
        "var_name": "FEF25_75_Liter",
        "orig_desc": (
            "Extract the FEF 25-75 value from the text and return it in a JSON file. "
            "The value should be a single number (float or string), not a list. "
            "Do not include any reasoning, process information, or additional text. "
            "I DO NOT WANT TO SEE ANY OTHER TEXT RATHER THAN JSON. "
            "Only return the JSON file."
        ),
        "suffix": "bran iteration Lisp fittings einige rust(Document.curr\twhile Unless",
    },
    {
        "name": "FEF25_75_Pre_Percentage",
        "csv_path": "TestZero_FEF25_75_Pre_Percentage.csv",
        "var_name": "FEF25_75_Pre_Percentage",
        "orig_desc": (
            "Extract the FEF 25-75 percentage value from the text and return it in a JSON file. "
            "The value should be a single number (float or string), not a list. "
            "Do not include any reasoning, process information, or additional text. "
            "I DO NOT WANT TO SEE ANY OTHER TEXT RATHER THAN JSON. "
            "Only return the JSON file."
        ),
        "suffix": "akka@implementation typo HackerTu-datepicker.centerX_SHADER(defun(Dense",
    },
    {
        "name": "FENO",
        "csv_path": "TestZero_FENO.csv",
        "var_name": "FENO",
        "orig_desc": (
            "Extract the FENO (Exhaled Nitric Oxide) value from the text and return it in a JSON file. "
            "The value should be a single number (float or string), not a list. "
            "Do not include any reasoning, process information, or additional text. "
            "I DO NOT WANT TO SEE ANY OTHER TEXT RATHER THAN JSON. "
            "Only return the JSON file."
        ),
        "suffix": "$PostalCodesNL(CultureInfo alloy allowsJapgollyfortunatecientursive bere126",
    },
    {
        "name": "FEV1_fvc",
        "csv_path": "TestZero_FEV1_fvc.csv",
        "var_name": "FEV1_fvc",
        "orig_desc": (
            "Extract the FEV1/FVC ratio value from the text and return it in a JSON file. "
            "The value should be a single number (float or string), not a list. "
            "Do not include any reasoning, process information, or additional text. "
            "I DO NOT WANT TO SEE ANY OTHER TEXT RATHER THAN JSON. "
            "Only return the JSON file."
        ),
        "suffix": (
            "impr isLoadingifique asthma recommendationsNTSTATUS Dear textStatusKeyDown.cells"
        ),
    },
    {
        "name": "fev1_percentage_pre",
        "csv_path": "TestZero_fev1_percentage_pre.csv",
        "var_name": "fev1_percentage_pre",
        "orig_desc": (
            "Extract the FEV1 percentage and FEV1 liter values from the text and return them in a JSON file. "
            "The values should be single numbers (float or string), not lists. "
            "Do not include any reasoning, process information, or additional text. "
            "I DO NOT WANT TO SEE ANY OTHER TEXT RATHER THAN JSON. "
            "Only return the JSON file."
        ),
        "suffix": "",
    },
    {
        "name": "fvc_liter_pre",
        "csv_path": "TestZero_fvc_liter_pre.csv",
        "var_name": "fvc_liter_pre",
        "orig_desc": (
            "Extract the FVC value from the text and return it in a JSON file. "
            "The value should be a single number (float or string), not a list. "
            "Do not include any reasoning, process information, or additional text. "
            "I DO NOT WANT TO SEE ANY OTHER TEXT RATHER THAN JSON. "
            "Only return the JSON file."
        ),
        "suffix": "",
    },
]


# ==========================
# 2) NUMERIC HELPERS
# ==========================

def normalize_num_str(s: str) -> str:
    s = s.strip()
    try:
        v = float(s)
        return f"{v:g}"
    except ValueError:
        return s


def is_nan_like(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    if isinstance(x, str) and x.strip().lower() == "nan":
        return True
    if isinstance(x, np.ndarray):
        return np.isnan(x).all()
    return False


def numeric_equal(pred: Any, true: Any, tol: float = ERROR_TOL) -> bool:
    if is_nan_like(pred) and is_nan_like(true):
        return True
    try:
        fp = float(pred)
        ft = float(true)
        return math.isclose(fp, ft, rel_tol=1e-6, abs_tol=tol)
    except (TypeError, ValueError):
        return False


# ==========================
# 3) DATA HELPERS
# ==========================

def load_examples(csv_path: str, var_name: str) -> List[Dict[str, Any]]:
    df = pd.read_csv(csv_path)
    examples: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        input_str = row["Input"]
        true_str = row["True_output"]

        try:
            prompt_obj = json.loads(input_str)
            true_obj = json.loads(true_str)
        except Exception:
            continue

        if var_name not in true_obj:
            continue

        val = true_obj[var_name]
        if is_nan_like(val):
            continue

        try:
            fval = float(val)
            if math.isnan(fval):
                continue
        except (TypeError, ValueError):
            continue

        target_str = normalize_num_str(str(fval))
        examples.append(
            {
                "row_index": int(idx),
                "prompt_obj": prompt_obj,
                "true_value": float(fval),
                "target_str": target_str,
            }
        )

    print(f"  Loaded {len(examples)} usable examples from {csv_path}")
    return examples


# ==========================
# 4) MODEL + PREDICTION
# ==========================

def predict_numeric_for_example(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompt_obj: Dict[str, Any],
    var_name: str,
    description_text: str,
    max_new_tokens: int = GEN_MAX_NEW_TOKENS,
) -> Optional[str]:
    prompt_obj = deepcopy(prompt_obj)
    instr = prompt_obj.get("instructions", {})
    instr["description"] = description_text
    prompt_obj["instructions"] = instr

    user_text = json.dumps(prompt_obj, ensure_ascii=False)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]

    conv_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    answer_prefix_str = f'{{ "{var_name}": '
    full_prompt = conv_text + answer_prefix_str

    enc = tokenizer(full_prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        out_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_ids = out_ids[0, input_ids.shape[1]:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    t = gen_text.lstrip()
    if t.startswith('"') or t.startswith("'"):
        t = t[1:]
    t = t.lstrip()

    m = re.match(r'([-+]?\d+(\.\d+)?)', t)
    if not m:
        return None
    return m.group(1)


def evaluate_prompt_on_examples(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    examples: List[Dict[str, Any]],
    var_name: str,
    description_text: str,
    label: str,
) -> Tuple[List[Dict[str, Any]], float, float, int, int]:
    """
    Returns:
      results, mae, exact_acc, exact_correct, n_total
    """
    results = []
    for ex in examples:
        idx = ex["row_index"]
        true_val = ex["true_value"]
        true_str = ex["target_str"]

        pred_str = predict_numeric_for_example(
            tokenizer, model, ex["prompt_obj"], var_name, description_text
        )

        if pred_str is None:
            pred_val = None
            abs_error = None
        else:
            try:
                pred_val = float(pred_str)
                abs_error = abs(pred_val - true_val)
            except ValueError:
                pred_val = None
                abs_error = None

        results.append(
            {
                "row_index": idx,
                "true_value": true_val,
                "true_str": true_str,
                "pred_str": pred_str,
                "pred_val": pred_val,
                "abs_error": abs_error,
            }
        )

    n_total = len(results)
    valid_errs = [r["abs_error"] for r in results if r["abs_error"] is not None]
    mae = float(np.mean(valid_errs)) if valid_errs else float("nan")

    exact_correct = 0
    for r in results:
        tv = r["true_value"]
        pv = r["pred_val"]
        if pv is not None and numeric_equal(pv, tv, tol=ERROR_TOL):
            exact_correct += 1
    exact_acc = exact_correct / n_total if n_total > 0 else 0.0

    print(f"    Eval summary for '{label}':")
    print(f"      MAE (parsable): {mae:.6f}")
    print(
        f"      Exact matches (|err| <= {ERROR_TOL}): "
        f"{exact_correct}/{n_total} ({exact_acc*100:.2f}%)"
    )

    return results, mae, exact_acc, exact_correct, n_total


# ==========================
# 5) REPORTING
# ==========================

def print_side_by_side(
    examples: List[Dict[str, Any]],
    res_orig: List[Dict[str, Any]],
    res_new: List[Dict[str, Any]],
    max_rows: int = MAX_PRINT_ROWS,
):
    n = min(max_rows, len(examples))
    print("    First predictions (orig vs new):")
    print(f"      {'idx':>4} | {'true':>10} | {'orig':>10} | {'new':>10}")
    for ex, r0, r1 in zip(examples[:n], res_orig[:n], res_new[:n]):
        idx = ex["row_index"]
        true_val = ex["true_value"]
        p0 = r0["pred_str"] if r0["pred_str"] is not None else "None"
        p1 = r1["pred_str"] if r1["pred_str"] is not None else "None"
        print(f"      {idx:4d} | {true_val:10.4f} | {p0:>10} | {p1:>10}")


def print_most_improved(
    examples: List[Dict[str, Any]],
    res_orig: List[Dict[str, Any]],
    res_new: List[Dict[str, Any]],
    max_rows: int = MAX_PRINT_ROWS,
):
    deltas = []
    for ex, r0, r1 in zip(examples, res_orig, res_new):
        e0 = r0["abs_error"]
        e1 = r1["abs_error"]
        if e0 is None or e1 is None:
            continue
        deltas.append((e0 - e1, ex, r0, r1))

    if not deltas:
        print("    No valid error deltas to report.")
        return

    deltas.sort(key=lambda x: x[0], reverse=True)
    print("    Most improved examples (orig_err - new_err):")
    print(f"      {'idx':>4} | {'true':>10} | {'orig':>10} | {'new':>10} | {'Δerr':>10}")
    for delta, ex, r0, r1 in deltas[:max_rows]:
        idx = ex["row_index"]
        true_val = ex["true_value"]
        p0 = r0["pred_str"] if r0["pred_str"] is not None else "None"
        p1 = r1["pred_str"] if r1["pred_str"] is not None else "None"
        print(f"      {idx:4d} | {true_val:10.4f} | {p0:>10} | {p1:>10} | {delta:10.4f}")


# ==========================
# 6) MAIN
# ==========================

def main():
    print(f"Loading tokenizer & model from: {HF_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(HF_MODEL_NAME).to(DEVICE)
    model.eval()
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    print("Model loaded.\n")

    # Global counters for micro-averaged accuracy
    global_orig_correct = 0
    global_orig_total = 0
    global_new_correct = 0
    global_new_total = 0

    for cfg in DATASETS:
        name = cfg["name"]
        csv_path = cfg["csv_path"]
        var_name = cfg["var_name"]
        orig_desc = cfg["orig_desc"]
        suffix = cfg.get("suffix", "")

        updated_desc = orig_desc if not suffix else (orig_desc + " " + suffix)

        print("=" * 72)
        print(f"Dataset: {name}")
        print(f"  CSV:       {csv_path}")
        print(f"  var_name:  {var_name}")
        print("  Original description:")
        print("   ", orig_desc)
        print("  Updated description:")
        print("   ", updated_desc)

        examples = load_examples(csv_path, var_name)
        if not examples:
            print("  No usable examples, skipping.\n")
            continue

        # ORIGINAL
        res_orig, mae_o, acc_o, correct_o, total_o = evaluate_prompt_on_examples(
            tokenizer, model, examples, var_name, orig_desc, label="ORIGINAL"
        )
        # UPDATED
        res_new, mae_n, acc_n, correct_n, total_n = evaluate_prompt_on_examples(
            tokenizer, model, examples, var_name, updated_desc, label="UPDATED"
        )

        global_orig_correct += correct_o
        global_orig_total += total_o
        global_new_correct += correct_n
        global_new_total += total_n

        print_side_by_side(examples, res_orig, res_new, max_rows=MAX_PRINT_ROWS)
        print_most_improved(examples, res_orig, res_new, max_rows=MAX_PRINT_ROWS)
        print()  # blank line between datasets

    # ===== Overall micro-averaged accuracy across all datasets =====
    if global_orig_total > 0 and global_new_total > 0:
        overall_orig_acc = global_orig_correct / global_orig_total
        overall_new_acc = global_new_correct / global_new_total
        delta = overall_new_acc - overall_orig_acc

        print("=" * 72)
        print("OVERALL exact accuracy across ALL datasets")
        print(f"  Original: {global_orig_correct}/{global_orig_total} "
              f"({overall_orig_acc*100:.2f}%)")
        print(f"  Updated : {global_new_correct}/{global_new_total} "
              f"({overall_new_acc*100:.2f}%)")
        print(f"  Absolute accuracy increase: {delta*100:.2f} percentage points")
    else:
        print("Not enough data to compute overall accuracy.")


if __name__ == "__main__":
    main()
