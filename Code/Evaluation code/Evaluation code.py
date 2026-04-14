import pandas as pd
import ast
import re
from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F
import sys
import numpy as np
from pathlib import Path


# =========================================================
# Text alignment: strip wrapping quotes + lowercase
# =========================================================
def strip_wrapping_quotes(s: str) -> str:
    """
    Remove only the outermost pair of quotes, including Chinese and English
    quote styles. Examples:
      “abc” -> abc
      "abc" -> abc
      ‘abc’ -> abc
      'abc' -> abc
    If the whole string is not wrapped, leave it unchanged.
    """
    if s is None:
        return ""
    s = str(s).strip()

    pairs = [
        ("“", "”"),
        ("‘", "’"),
        ('"', '"'),
        ("'", "'"),
    ]

    changed = True
    while changed:  # Allow nested wrapping such as ""“abc”"".
        changed = False
        for lq, rq in pairs:
            if len(s) >= 2 and s.startswith(lq) and s.endswith(rq):
                s = s[1:-1].strip()
                changed = True
                break
    return s


def normalize_text_for_align(s: str) -> str:
    """
    Normalize the Text field for alignment before merging:
    - remove wrapping quotes
    - lowercase the string to preserve the original evaluation behavior
    """
    return strip_wrapping_quotes(s).lower()


# =========================================================
# Text alignment statistics
# =========================================================
def report_text_alignment(file1: pd.DataFrame, file2: pd.DataFrame, merged_df: pd.DataFrame) -> None:
    """
    Report how many predicted sentences align with the ground-truth sentences
    based only on Text alignment, regardless of whether Relations are NaN or
    empty.
    - exact_key_match: size of the normalized Text intersection
    - truth_only: number of Text entries that appear only in truth
    - pred_only: number of Text entries that appear only in predictions

    Also report:
    - both_have_relations: number of sentences where Relations is non-NaN on
      both sides, which matches the previous matched_sentences definition
    """
    truth_keys = set(file1['Text'].dropna().unique())
    pred_keys = set(file2['Text'].dropna().unique())

    inter = truth_keys & pred_keys
    truth_only = truth_keys - pred_keys
    pred_only = pred_keys - truth_keys

    print(f"[TextAlign] exact_key_match={len(inter)} / truth_unique_text={len(truth_keys)} / pred_unique_text={len(pred_keys)}")
    print(f"[TextAlignDetail] truth_only={len(truth_only)}, pred_only={len(pred_only)}")

    both_have_relations_df = merged_df[pd.notna(merged_df['Relations_x']) & pd.notna(merged_df['Relations_y'])]
    both_have_relations = both_have_relations_df['Text'].nunique()
    print(f"[TextBothHaveRelations] sentences={both_have_relations}")


# =========================================================
# Relations / triplet parsing and matching
# =========================================================
def process_relation_string(relation_str):
    if pd.isna(relation_str):
        return ''
    try:
        relation_list = ast.literal_eval(relation_str)
        processed_parts = []
        for relation in relation_list:
            parts = relation.split(': ')
            if len(parts) > 1:
                processed_parts.append(parts[1].strip())
        return '; '.join(processed_parts)
    except (ValueError, SyntaxError):
        return relation_str


def parse_triplets(relation_str):
    triplet_pattern = r"\(([^,]+),\s*([^,]+),\s*([^\)]+)\)"
    return re.findall(triplet_pattern, relation_str)


def match_triplets(triplets_x, triplets_y):
    result = []
    for t1 in triplets_x:
        head1, rel1, tail1 = [x.lower().replace("(", "").replace(")", "") for x in t1]
        for t2 in triplets_y:
            head2, rel2, tail2 = [x.lower().replace("(", "").replace(")", "") for x in t2]
            if (head1 in head2 and tail1 in tail2) or (head2 in head1 and tail2 in tail1):
                result.append(f"({', '.join(t1)})")
                result.append(f"({', '.join(t2)})")
    final_result = []
    for i in range(0, len(result), 2):
        final_result.append(f"{{{result[i]}, {result[i + 1]}}}")
    return "; ".join(final_result)


def process_and_match_triplets(file1_path, file2_path):
    file1 = pd.read_csv(file1_path, header=None, names=['Text', 'Relations'])
    file2 = pd.read_csv(file2_path, header=None, names=['Text', 'Relations'])

    file1['Text'] = file1['Text'].str.lower()
    file2['Text'] = file2['Text'].str.lower()

    merged_df = pd.merge(file1, file2, on='Text', how='outer')

    # Print sentence alignment based only on Text.
    report_text_alignment(file1, file2, merged_df)

    # Preprocess the prediction-side Relations column.
    merged_df['Relations_y'] = merged_df['Relations_y'].apply(process_relation_string)

    def count_triplets_in_column(df, column_name):
        return df[column_name].apply(lambda x: len(parse_triplets(x)) if pd.notna(x) else 0)

    merged_df['Relations_x_triplet_count'] = count_triplets_in_column(merged_df, 'Relations_x')
    merged_df['Relations_y_triplet_count'] = count_triplets_in_column(merged_df, 'Relations_y')

    truth_triples = merged_df['Relations_x_triplet_count'].sum()
    extracted_triples = merged_df['Relations_y_triplet_count'].sum()

    final_list = []
    for _, row in merged_df.iterrows():
        if pd.notna(row['Relations_x']) and pd.notna(row['Relations_y']):
            triplets_x = parse_triplets(str(row['Relations_x']).lower())
            triplets_y = parse_triplets(str(row['Relations_y']).lower())
            output = match_triplets(triplets_x, triplets_y)

            output_split = output.split(';')
            final_list.extend([item.strip() for item in output_split if item.strip()])

    return final_list, truth_triples, extracted_triples


# =========================================================
# BERT similarity
# =========================================================
def initialize_bert_model(model_file='../bert_model'):
    tokenizer = BertTokenizer.from_pretrained(model_file, clean_up_tokenization_spaces=True)
    model = BertModel.from_pretrained(model_file)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    return tokenizer, model, device


def get_embedding(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]


def cosine_similarity(vec1, vec2):
    return F.cosine_similarity(vec1, vec2)


def calculate_similarity_for_triplets(triples_list, tokenizer, model, device, threshold):
    similarities = []
    num = 0

    for idx, triple_pair in enumerate(triples_list):
        try:
            triple_pair = triple_pair.strip('{}').split('), (')
            if len(triple_pair) != 2:
                raise ValueError(f"Malformed triple pair at index {idx}: {triple_pair}")

            triple1 = triple_pair[0].lower().replace('(', '').replace(')', '').split(', ')
            triple2 = triple_pair[1].lower().replace('(', '').replace(')', '').split(', ')

            triple1_text = triple1[1].lower()
            triple2_text = triple2[1].lower()

            triple1_embedding = get_embedding(triple1_text, tokenizer, model, device)
            triple2_embedding = get_embedding(triple2_text, tokenizer, model, device)

            similarity = cosine_similarity(triple1_embedding, triple2_embedding).item()

            if similarity >= threshold:
                num += 1

            similarities.append((triple1_text, triple2_text, similarity))

        except Exception as e:
            print(f"Error processing triple pair at index {idx}: {e}")
            continue

    return similarities, num


# =========================================================
# Entity-match statistics
# =========================================================
def count_entity_matched(final_list):
    pair_cnt = len(final_list)

    truth_set = set()
    extract_set = set()

    for s in final_list:
        if not s:
            continue
        s = s.strip()

        triplets = re.findall(r"\([^)]*\)", s)
        if len(triplets) >= 2:
            truth_set.add(triplets[0].strip())
            extract_set.add(triplets[1].strip())

    return pair_cnt, len(truth_set), len(extract_set)


# =========================================================
# Single-method evaluation (original main entry)
# =========================================================
def main(file1_path, file2_path, threshold):
    final_list, truth_num, extract_num = process_and_match_triplets(file1_path, file2_path)
    final_list = [item for item in final_list if item != '']

    pair_cnt, truth_u_cnt, extract_u_cnt = count_entity_matched(final_list)
    print(f"[EntityMatch] pairs={pair_cnt}, truth_unique={truth_u_cnt}, extract_unique={extract_u_cnt}")

    tokenizer, model, device = initialize_bert_model()

    similarities, num = calculate_similarity_for_triplets(final_list, tokenizer, model, device, threshold)

    precision = round(num / extract_num, 2)
    recall = round(num / truth_num, 2)
    f1 = round((2 * precision * recall) / (precision + recall), 2)

    print(num)
    print(extract_num)
    print(truth_num)
    print(f"Threshold {threshold}: Precision {precision} Recall {recall} F1 {f1}")


# =========================================================
# Added: permutation-test p-value + bootstrap CI for comparing the F1
# difference between two methods.
# =========================================================
def micro_f1_from_counts(tp: int, pred_total: int, gold_total: int) -> float:
    denom = pred_total + gold_total
    return (2.0 * tp / denom) if denom > 0 else 0.0


MODEL_ORDER = {"4o": 0, "gemini": 1, "mini": 2, "llama": 3, "claude": 4}


def canonical_model_name(name_or_path: str):
    s = str(name_or_path).lower().replace(" ", "")
    if "gpt-4o" in s or re.search(r"(^|[^a-z0-9])4o([^a-z0-9]|$)", s):
        return "4o"
    if "gemini" in s:
        return "gemini"
    if "mini" in s:
        return "mini"
    if "llama" in s:
        return "llama"
    if "claude" in s:
        return "claude"
    return None


def model_sort_key(model_name: str):
    return (MODEL_ORDER.get(model_name, 99), model_name)


def choose_relation_column(df: pd.DataFrame, text_idx=0):
    candidate_cols = [c for c in df.columns if c != text_idx]
    if not candidate_cols:
        return text_idx

    best_col = candidate_cols[0]
    best_score = (-1, -1, -1, -1)

    for c in candidate_cols:
        series = df[c].fillna("").astype(str).str.strip()
        non_empty = int((series != "").sum())
        triplet_like = int(series.str.contains(r"\([^,]+,\s*[^,]+,\s*[^\)]+\)", regex=True).sum())
        list_like = int(series.str.contains(r"\[[^\]]*\]", regex=True).sum())
        score = (triplet_like, list_like, non_empty, int(c))
        if score > best_score:
            best_score = score
            best_col = c

    return best_col


def load_text_relations_csv(file_path: str) -> pd.DataFrame:
    # Keep the exact CSV-reading behavior used in bert_cal.py.
    df = pd.read_csv(file_path, header=None, names=["Text", "Relations"])
    df["Text"] = df["Text"].str.lower()
    return df


def build_truth_keys(file1_path: str):
    file1 = load_text_relations_csv(file1_path)
    keys = list(dict.fromkeys(file1["Text"].dropna().tolist()))
    return keys


def build_per_text_stats_against_gt(file1_path: str, file2_path: str, threshold: float,
                                   tokenizer, model, device):
    file1 = load_text_relations_csv(file1_path)
    file2 = load_text_relations_csv(file2_path)

    merged_df = pd.merge(file1, file2, on="Text", how="outer")
    merged_df["Relations_y"] = merged_df["Relations_y"].apply(process_relation_string)

    def count_triplets_in_cell(cell):
        if pd.isna(cell):
            return 0
        return len(parse_triplets(str(cell)))

    stats = {}

    for _, row in merged_df.iterrows():
        text = row["Text"]
        if pd.isna(text):
            continue

        gold_cnt = count_triplets_in_cell(row["Relations_x"])
        pred_cnt = count_triplets_in_cell(row["Relations_y"])

        tp_i = 0
        if pd.notna(row["Relations_x"]) and pd.notna(row["Relations_y"]):
            triplets_x = parse_triplets(str(row["Relations_x"]).lower())
            triplets_y = parse_triplets(str(row["Relations_y"]).lower())

            output = match_triplets(triplets_x, triplets_y)
            pair_strs = [s.strip() for s in output.split(";") if s.strip()]

            if pair_strs:
                _, num = calculate_similarity_for_triplets(pair_strs, tokenizer, model, device, threshold)
                tp_i = int(num)

        old_tp, old_pred, old_gold = stats.get(text, (0, 0, 0))
        stats[text] = (old_tp + int(tp_i), old_pred + int(pred_cnt), old_gold + int(gold_cnt))

    return stats


def aggregate_counts(keys, stats):
    tp = pred_total = gold_total = 0
    for k in keys:
        tpi, pi, gi = stats.get(k, (0, 0, 0))
        tp += int(tpi)
        pred_total += int(pi)
        gold_total += int(gi)
    return tp, pred_total, gold_total


def aggregate_counts_all(stats):
    tp = pred_total = gold_total = 0
    for tpi, pi, gi in stats.values():
        tp += int(tpi)
        pred_total += int(pi)
        gold_total += int(gi)
    return tp, pred_total, gold_total


def paired_permutation_test_delta_micro_f1(keys, stats_a, stats_b, n_perm=2000, seed=0):
    """
    paired permutation test (two-sided) for Δmicro-F1 = F1(A) - F1(B)
    For each Text, swap the contribution of A and B with 50% probability.
    """
    rng = np.random.default_rng(seed)

    tp_a, pred_a, gold_a = aggregate_counts(keys, stats_a)
    tp_b, pred_b, gold_b = aggregate_counts(keys, stats_b)

    f1_a = micro_f1_from_counts(tp_a, pred_a, gold_a)
    f1_b = micro_f1_from_counts(tp_b, pred_b, gold_b)
    delta_obs = f1_a - f1_b

    exceed = 0
    for _ in range(n_perm):
        tp1 = pred1 = gold1 = 0
        tp2 = pred2 = gold2 = 0

        for k in keys:
            a = stats_a.get(k, (0, 0, 0))
            b = stats_b.get(k, (0, 0, 0))

            if rng.random() < 0.5:
                t1, p1, g1 = a
                t2, p2, g2 = b
            else:
                t1, p1, g1 = b
                t2, p2, g2 = a

            tp1 += t1; pred1 += p1; gold1 += g1
            tp2 += t2; pred2 += p2; gold2 += g2

        d = micro_f1_from_counts(tp1, pred1, gold1) - micro_f1_from_counts(tp2, pred2, gold2)
        if abs(d) >= abs(delta_obs):
            exceed += 1

    p_value = (exceed + 1) / (n_perm + 1)  # add-one smoothing
    return f1_a, f1_b, delta_obs, p_value


def paired_bootstrap_ci_delta_micro_f1(keys, stats_a, stats_b, n_boot=2000, seed=0, alpha=0.05):
    """
    Paired bootstrap confidence interval for delta micro-F1 using resampling
    with replacement over Text items.
    """
    rng = np.random.default_rng(seed)
    keys_arr = np.array(keys, dtype=object)
    n = len(keys_arr)

    deltas = []
    for _ in range(n_boot):
        sample = rng.choice(keys_arr, size=n, replace=True)

        tp_a = pred_a = gold_a = 0
        tp_b = pred_b = gold_b = 0

        for k in sample:
            t1, p1, g1 = stats_a.get(k, (0, 0, 0))
            t2, p2, g2 = stats_b.get(k, (0, 0, 0))
            tp_a += t1; pred_a += p1; gold_a += g1
            tp_b += t2; pred_b += p2; gold_b += g2

        d = micro_f1_from_counts(tp_a, pred_a, gold_a) - micro_f1_from_counts(tp_b, pred_b, gold_b)
        deltas.append(d)

    deltas = np.array(deltas, dtype=float)
    low = float(np.quantile(deltas, alpha / 2))
    high = float(np.quantile(deltas, 1 - alpha / 2))
    return low, high


def compare_two_methods_with_stats(file1_path, fileA_path, fileB_path, threshold,
                                  n_perm=2000, n_boot=2000, seed=0, model_file='../bert_model'):
    return compare_two_methods_with_stats_fast(
        file1_path=file1_path,
        fileA_path=fileA_path,
        fileB_path=fileB_path,
        threshold=threshold,
        n_perm=n_perm,
        n_boot=n_boot,
        seed=seed,
        model_file=model_file,
    )


def compute_prf_from_stats(stats):
    tp, pred_total, gold_total = aggregate_counts_all(stats)
    precision = round(tp / pred_total, 2) if pred_total > 0 else 0.0
    recall = round(tp / gold_total, 2) if gold_total > 0 else 0.0
    f1 = round((2 * precision * recall) / (precision + recall), 2) if (precision + recall) > 0 else 0.0
    return tp, pred_total, gold_total, precision, recall, f1


def print_significance_result(keys, stats_a, stats_b, label_a, label_b, threshold,
                              n_perm=2000, n_boot=2000, seed=0):
    f1_a, f1_b, delta, pval = paired_permutation_test_delta_micro_f1(
        keys, stats_a, stats_b, n_perm=n_perm, seed=seed
    )
    ci_low, ci_high = paired_bootstrap_ci_delta_micro_f1(
        keys, stats_a, stats_b, n_boot=n_boot, seed=seed, alpha=0.05
    )

    print("\n" + "-" * 60)
    print(f"[Stats] Compare @ thr={threshold}")
    print(f"[Stats] A={label_a}")
    print(f"[Stats] B={label_b}")
    print(f"[Stats] F1_A={f1_a:.4f}  F1_B={f1_b:.4f}  ΔF1(A-B)={delta:.4f}")
    print(f"[PermutationTest] n_perm={n_perm}  p_value(two-sided)={pval:.6f}")
    print(f"[BootstrapCI] n_boot={n_boot}  95% CI for ΔF1 = [{ci_low:.4f}, {ci_high:.4f}]")
    print("-" * 60)


def compare_two_methods_with_stats_fast(file1_path, fileA_path, fileB_path, threshold,
                                        n_perm=2000, n_boot=2000, seed=0, model_file='../bert_model'):
    tokenizer, model, device = initialize_bert_model(model_file=model_file)
    keys = build_truth_keys(file1_path)
    stats_a = build_per_text_stats_against_gt(file1_path, fileA_path, threshold, tokenizer, model, device)
    stats_b = build_per_text_stats_against_gt(file1_path, fileB_path, threshold, tokenizer, model, device)
    print_significance_result(
        keys, stats_a, stats_b,
        label_a=fileA_path, label_b=fileB_path,
        threshold=threshold, n_perm=n_perm, n_boot=n_boot, seed=seed
    )


def resolve_groundtruth_path():
    candidates = [
        Path("groundtruth") / "The groundtruth for test API dataset.csv.csv",
        Path("The groundtruth for test API dataset.csv.csv"),
        Path("Docs2KG") / "The groundtruth for test API dataset.csv.csv",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError("Cannot find groundtruth file: The groundtruth for test API dataset.csv.csv")


def collect_dhtw_model_files():
    out = {}
    for p in sorted(Path("DHTW").glob("*.csv")):
        model = canonical_model_name(p.name)
        if model:
            out[model] = str(p)
    return out


def collect_docs2kg_model_files():
    out = {}
    root = Path("Docs2KG")
    if not root.exists():
        return out
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.lower() == "mul":
            continue
        p = d / "2.final_output.csv"
        if p.exists():
            model = canonical_model_name(d.name)
            if model:
                out[model] = str(p)
    return out


def collect_docs2kg_mul_model_files():
    out = {}
    root = Path("Docs2KG") / "Mul"
    if not root.exists():
        return out
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        p = d / "2.final_output.csv"
        if p.exists():
            model = canonical_model_name(d.name)
            if model:
                out[model] = str(p)
    return out


def collect_edc_model_files():
    out = {}
    root = Path("EDC")
    if not root.exists():
        return out
    for p in sorted(root.glob("*.csv")):
        model = canonical_model_name(p.name)
        if model:
            out[model] = str(p)
    return out


def collect_csv_files(folder_name: str):
    p = Path(folder_name)
    if not p.exists():
        return []
    return [str(x) for x in sorted(p.glob("*.csv"))]


def add_eval_item(eval_items, seen_paths, label, file_path):
    normalized = str(Path(file_path))
    if normalized in seen_paths:
        return
    seen_paths.add(normalized)
    eval_items.append((label, normalized))


def build_global_eval_keys(file1_path, eval_items):
    ordered = {}

    file1 = load_text_relations_csv(file1_path)
    for t in file1["Text"].dropna().tolist():
        ordered.setdefault(t, None)

    for _, p in eval_items:
        df = load_text_relations_csv(p)
        for t in df["Text"].dropna().tolist():
            ordered.setdefault(t, None)

    return list(ordered.keys())


def compute_metrics_bert_style(file1_path, file2_path, threshold, model_file="../bert_model"):
    tokenizer, model, device = initialize_bert_model(model_file=model_file)
    final_list, truth_num, extract_num = process_and_match_triplets(file1_path, file2_path)
    final_list = [item for item in final_list if item != ""]
    _, num = calculate_similarity_for_triplets(final_list, tokenizer, model, device, threshold)

    if extract_num == 0 or truth_num == 0:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        precision = round(num / extract_num, 2)
        recall = round(num / truth_num, 2)
        f1 = round((2 * precision * recall) / (precision + recall), 2) if (precision + recall) else 0.0

    return int(num), int(extract_num), int(truth_num), float(precision), float(recall), float(f1)


# =========================================================
# Entry point: automatically collect files by directory, then run metrics and
# significance tests.
# =========================================================
if __name__ == '__main__':
    print("PY:", sys.executable)
    print("VER:", sys.version)
    print("PATH0:", sys.path[0])

    thresholds = [0.90, 0.92, 0.94]
    n_perm = 2000
    n_boot = 2000
    seed = 0
    model_file = "../bert_model"

    file1_path = resolve_groundtruth_path()
    print(f"[GroundTruth] {file1_path}")

    dhtw_map = collect_dhtw_model_files()
    docs2kg_map = collect_docs2kg_model_files()
    docs2kg_mul_map = collect_docs2kg_mul_model_files()
    edc_map = collect_edc_model_files()
    kw_files = collect_csv_files("KW&H")
    ablation_files = collect_csv_files("ablation")
    iteration_files = collect_csv_files("iteration")

    if not dhtw_map:
        raise FileNotFoundError("No model file found under DHTW/*.csv")

    tokenizer, model, device = initialize_bert_model(model_file=model_file)
    stats_cache = {}
    metric_cache = {}

    def get_stats_cached(file_path: str, threshold: float):
        k = (file_path, threshold)
        if k not in stats_cache:
            stats_cache[k] = build_per_text_stats_against_gt(
                file1_path, file_path, threshold, tokenizer, model, device
            )
        return stats_cache[k]

    def get_metrics_cached(file_path: str, threshold: float):
        k = (file_path, threshold)
        if k not in metric_cache:
            metric_cache[k] = compute_metrics_bert_style(
                file1_path, file_path, threshold, model_file=model_file
            )
        return metric_cache[k]

    eval_items = []
    seen_paths = set()

    for m in sorted(dhtw_map.keys(), key=model_sort_key):
        add_eval_item(eval_items, seen_paths, f"DHTW[{m}]", dhtw_map[m])
    for m in sorted(docs2kg_map.keys(), key=model_sort_key):
        add_eval_item(eval_items, seen_paths, f"Docs2KG[{m}]", docs2kg_map[m])
    for m in sorted(docs2kg_mul_map.keys(), key=model_sort_key):
        add_eval_item(eval_items, seen_paths, f"Docs2KG_Mul[{m}]", docs2kg_mul_map[m])
    for m in sorted(edc_map.keys(), key=model_sort_key):
        add_eval_item(eval_items, seen_paths, f"EDC[{m}]", edc_map[m])
    for p in kw_files:
        add_eval_item(eval_items, seen_paths, f"KW&H[{Path(p).stem}]", p)
    for p in ablation_files:
        add_eval_item(eval_items, seen_paths, f"Ablation[{Path(p).stem}]", p)
    for p in iteration_files:
        add_eval_item(eval_items, seen_paths, f"Iteration[{Path(p).stem}]", p)

    keys = build_global_eval_keys(file1_path, eval_items)

    print("\n================= Metrics For All Files =================")
    for label, path in eval_items:
        print(f"\n######## {label} ########")
        for thr in thresholds:
            tp, pred_total, gold_total, precision, recall, f1 = get_metrics_cached(path, thr)
            print(
                f"[Metric] thr={thr} | TP={tp} Pred={pred_total} Gold={gold_total} "
                f"| Precision={precision:.2f} Recall={recall:.2f} F1={f1:.2f}"
            )

    print("\n================= Significance: Model-Aligned (DHTW vs Others) =================")
    aligned_methods = [
        ("Docs2KG", docs2kg_map),
        ("Docs2KG_Mul", docs2kg_mul_map),
        ("EDC", edc_map),
    ]
    for method_name, method_map in aligned_methods:
        shared_models = sorted(set(dhtw_map.keys()) & set(method_map.keys()), key=model_sort_key)
        if not shared_models:
            print(f"[Skip] No shared models between DHTW and {method_name}")
            continue
        print(f"\n###### DHTW vs {method_name} ######")
        for m in shared_models:
            label_a = f"DHTW[{m}]"
            label_b = f"{method_name}[{m}]"
            path_a = dhtw_map[m]
            path_b = method_map[m]
            for thr in thresholds:
                stats_a = get_stats_cached(path_a, thr)
                stats_b = get_stats_cached(path_b, thr)
                print_significance_result(
                    keys, stats_a, stats_b, label_a, label_b, thr,
                    n_perm=n_perm, n_boot=n_boot, seed=seed
                )

    ref_4o_path = dhtw_map.get("4o")
    if not ref_4o_path:
        sorted_models = sorted(dhtw_map.keys(), key=model_sort_key)
        ref_model = sorted_models[0]
        ref_4o_path = dhtw_map[ref_model]
        print(f"[Warn] DHTW 4o not found, fallback reference model={ref_model}")

    print("\n================= Significance: KW&H vs DHTW[4o] =================")
    for p in kw_files:
        label_a = f"KW&H[{Path(p).stem}]"
        label_b = "DHTW[4o]"
        for thr in thresholds:
            stats_a = get_stats_cached(p, thr)
            stats_b = get_stats_cached(ref_4o_path, thr)
            print_significance_result(
                keys, stats_a, stats_b, label_a, label_b, thr,
                n_perm=n_perm, n_boot=n_boot, seed=seed
            )

    print("\n================= Significance: Ablation vs DHTW[4o] =================")
    for p in ablation_files:
        label_a = f"Ablation[{Path(p).stem}]"
        label_b = "DHTW[4o]"
        for thr in thresholds:
            stats_a = get_stats_cached(p, thr)
            stats_b = get_stats_cached(ref_4o_path, thr)
            print_significance_result(
                keys, stats_a, stats_b, label_a, label_b, thr,
                n_perm=n_perm, n_boot=n_boot, seed=seed
            )

    print("\n================= Significance: Iteration vs DHTW[4o] =================")
    for p in iteration_files:
        label_a = f"Iteration[{Path(p).stem}]"
        label_b = "DHTW[4o]"
        for thr in thresholds:
            stats_a = get_stats_cached(p, thr)
            stats_b = get_stats_cached(ref_4o_path, thr)
            print_significance_result(
                keys, stats_a, stats_b, label_a, label_b, thr,
                n_perm=n_perm, n_boot=n_boot, seed=seed
            )
