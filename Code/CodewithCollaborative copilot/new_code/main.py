import csv
import yaml
from itertools import combinations
import kg_exploration as KE
import kg_construction as KC
import os
import json
from datetime import datetime

def _resolve_secret_reference(value, default=""):
    if value is None:
        return default

    raw = str(value).strip()
    if not raw:
        return default

    if raw.startswith("${") and raw.endswith("}"):
        env_name = raw[2:-1].strip()
        return os.getenv(env_name, default).strip()

    env_value = os.getenv(raw)
    if env_value:
        return env_value.strip()

    if raw.upper() == raw and "_" in raw:
        return default

    return raw


def resolve_runtime_config(universal_config):
    config = dict(universal_config or {})

    config["OpenAI_API_Base"] = (
        _resolve_secret_reference(config.get("OpenAI_API_Base"), default=os.getenv("OPENAI_API_BASE", "")).strip()
        or "https://api.openai.com/v1"
    )

    resolved_keys = [
        _resolve_secret_reference(item)
        for item in config.get("API_key_list", []) or []
    ]
    resolved_keys = [key for key in resolved_keys if key]

    legacy_key = _resolve_secret_reference(config.get("API_key"))
    if legacy_key and legacy_key not in resolved_keys:
        resolved_keys.append(legacy_key)

    if not resolved_keys:
        raise ValueError(
            "No API key is configured. Set OPENAI_API_KEY or reference another populated environment variable in config.yaml."
        )

    config["API_key_list"] = resolved_keys
    config["API_key"] = resolved_keys[0]
    return config

def print_total_statistics(ke_stats=None, kc_stats=None):
    print("\n" + "="*80)
    print("Overall Statistics Report")
    print("="*80)

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    total_llm_calls = 0
    all_response_times = []
    total_human_feedback_time = 0

    if ke_stats:
        total_prompt_tokens += ke_stats['total_prompt_tokens']
        total_completion_tokens += ke_stats['total_completion_tokens']
        total_tokens += ke_stats['total_tokens']
        total_llm_calls += ke_stats['llm_call_count']
        all_response_times.extend(ke_stats['response_times'])
        total_human_feedback_time += ke_stats['human_feedback_time']

    if kc_stats:
        total_prompt_tokens += kc_stats['total_prompt_tokens']
        total_completion_tokens += kc_stats['total_completion_tokens']
        total_tokens += kc_stats['total_tokens']
        total_llm_calls += kc_stats['llm_call_count']
        all_response_times.extend(kc_stats['response_times'])

    print("Token Statistics:")
    print(f"  - Total Prompt Tokens: {total_prompt_tokens:,}")
    print(f"  - Total Completion Tokens: {total_completion_tokens:,}")
    print(f"  - Total Tokens: {total_tokens:,}")

    print("\nTime Statistics:")
    print(f"  - Total LLM Calls: {total_llm_calls}")
    if all_response_times:
        print(f"  - Average Response Time: {sum(all_response_times)/len(all_response_times):.2f}s")
        print(f"  - Fastest Response Time: {min(all_response_times):.2f}s")
        print(f"  - Slowest Response Time: {max(all_response_times):.2f}s")
        print(f"  - Total LLM Response Time: {sum(all_response_times):.2f}s ({sum(all_response_times)/60:.2f} min)")
    if total_human_feedback_time > 0:
        print(f"  - Total Human Feedback Time: {total_human_feedback_time:.2f}s ({total_human_feedback_time/60:.2f} min)")

    if all_response_times and total_human_feedback_time > 0:
        total_time = sum(all_response_times) + total_human_feedback_time
        print(f"  - Total Time (LLM + Human Feedback): {total_time:.2f}s ({total_time/60:.2f} min)")

    print("="*80 + "\n")

    # Save overall statistics to a file
    save_total_statistics_to_file(ke_stats, kc_stats)

def save_total_statistics_to_file(ke_stats=None, kc_stats=None):
    """Save overall statistics to a file."""
    save_path = "../output/total_statistics.txt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    total_llm_calls = 0
    all_response_times = []
    total_human_feedback_time = 0

    if ke_stats:
        total_prompt_tokens += ke_stats['total_prompt_tokens']
        total_completion_tokens += ke_stats['total_completion_tokens']
        total_tokens += ke_stats['total_tokens']
        total_llm_calls += ke_stats['llm_call_count']
        all_response_times.extend(ke_stats['response_times'])
        total_human_feedback_time += ke_stats['human_feedback_time']

    if kc_stats:
        total_prompt_tokens += kc_stats['total_prompt_tokens']
        total_completion_tokens += kc_stats['total_completion_tokens']
        total_tokens += kc_stats['total_tokens']
        total_llm_calls += kc_stats['llm_call_count']
        all_response_times.extend(kc_stats['response_times'])
        total_human_feedback_time += kc_stats.get('human_feedback_time', 0)

    if kc_stats:
        total_prompt_tokens += kc_stats['total_prompt_tokens']
        total_completion_tokens += kc_stats['total_completion_tokens']
        total_tokens += kc_stats['total_tokens']
        total_llm_calls += kc_stats['llm_call_count']
        all_response_times.extend(kc_stats['response_times'])

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Overall Statistics Report\n")
        f.write(f"Generated At: {current_time}\n")
        f.write("="*80 + "\n")
        f.write("Token Statistics:\n")
        f.write(f"  - Total Prompt Tokens: {total_prompt_tokens:,}\n")
        f.write(f"  - Total Completion Tokens: {total_completion_tokens:,}\n")
        f.write(f"  - Total Tokens: {total_tokens:,}\n")

        f.write("\nTime Statistics:\n")
        f.write(f"  - Total LLM Calls: {total_llm_calls}\n")
        if all_response_times:
            f.write(f"  - Average Response Time: {sum(all_response_times)/len(all_response_times):.2f}s\n")
            f.write(f"  - Fastest Response Time: {min(all_response_times):.2f}s\n")
            f.write(f"  - Slowest Response Time: {max(all_response_times):.2f}s\n")
            f.write(f"  - Total LLM Response Time: {sum(all_response_times):.2f}s ({sum(all_response_times)/60:.2f} min)\n")
        if total_human_feedback_time > 0:
            f.write(f"  - Total Human Feedback Time: {total_human_feedback_time:.2f}s ({total_human_feedback_time/60:.2f} min)\n")

        if all_response_times and total_human_feedback_time > 0:
            total_time = sum(all_response_times) + total_human_feedback_time
            f.write(f"  - Total Time (LLM + Human Feedback): {total_time:.2f}s ({total_time/60:.2f} min)\n")

        f.write("="*80 + "\n")

    print(f"Overall statistics saved to: {save_path}")

def dedupe_entity_csv(entity_csv_path, max_cell_len=10_000_000):
    # Increase the CSV field size limit to avoid read errors.
    try:
        csv.field_size_limit(2_147_483_647)
    except OverflowError:
        csv.field_size_limit(50_000_000)

    rows = []
    skipped_cells = 0
    total_before = 0
    total_after = 0

    with open(entity_csv_path, encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            while len(row) < 3:
                row.append("")

            # Clear overlong cells to keep CSV parsing stable.
            for i in range(3):
                if row[i] is not None and len(row[i]) > max_cell_len:
                    row[i] = ""
                    skipped_cells += 1

            text = row[0]
            entity_str = row[1]

            entities = []
            seen = set()
            if entity_str:
                items = entity_str.split("; ")
                total_before += sum(1 for item in items if ": " in item)
                for item in items:
                    if ": " not in item:
                        continue
                    name, etype = item.split(": ", 1)
                    name = name.strip()
                    if not name or name in seen:
                        continue
                    seen.add(name)
                    etype = etype.strip().replace(";", "")
                    entities.append((name, etype))

            total_after += len(entities)

            if entities:
                entity_str_dedup = "; ".join([f"{n}: {t}" for n, t in entities])
                entity_names = [n for n, _ in entities]
                entity_pairs = list(combinations(entity_names, 2)) if len(entity_names) >= 2 else []
            else:
                entity_str_dedup = ""
                entity_pairs = []

            rows.append([text, entity_str_dedup, str(entity_pairs)])

    tmp_path = entity_csv_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    os.replace(tmp_path, entity_csv_path)

    removed = max(total_before - total_after, 0)
    if removed > 0 or skipped_cells > 0:
        print(f"[WARNING] Entity deduplication complete. Removed duplicates: {removed}, cleared overlong cells: {skipped_cells}")

def kg_exploration(universal_config):
    kgexplorer = KE.KGExploration(**universal_config)
    file_content_list = kgexplorer.read_seed_files()
    chunks = kgexplorer.split_list(file_content_list)
    all_entity_lists = []
    all_relation_lists = []
    all_entity_types = []
    all_relation_types = []


    for chunk in chunks:
        suggestions = kgexplorer.read_suggestions()
        chunk_list, entity_list = kgexplorer.process_entity_extraction(chunk, suggestions)
        kgexplorer.save_extracted_entities(chunk_list, entity_list)
        all_entity_lists.append(entity_list)


        chunk_list, entity_pair_list = kgexplorer.read_entity_infos()


        relation_triple_list = kgexplorer.process_relation_extraction(chunk_list, entity_pair_list, suggestions)
        kgexplorer.save_extracted_relations(chunk_list, entity_pair_list, relation_triple_list)
        all_relation_lists.append(relation_triple_list)


        entity_type_list = kgexplorer.process_entity_type_labeling(chunk_list, entity_list)
        kgexplorer.save_labeled_entity_types(chunk_list, entity_list, entity_type_list)
        all_entity_types.append(entity_type_list)

        unique_entity_type_list = kgexplorer.read_entity_types()
        entity_type_definitions, entity_types = kgexplorer.entity_type_fusion(unique_entity_type_list)
        kgexplorer.save_fused_entity_types(entity_type_definitions, entity_types)

        unique_relation_type_list, relation_instances = kgexplorer.read_relation_types()
        relation_types, relation_type_definitions, relation_instances, final_relation_subtypes = kgexplorer.relation_type_fusion(unique_relation_type_list, relation_instances)
        kgexplorer.save_fused_relation_types(relation_types, relation_type_definitions, relation_instances, final_relation_subtypes)
        schema = kgexplorer.generate_kg_schema()
        kgexplorer.save_kg_schema(schema)
        kgexplorer.pause_and_process_csv()
    
    # Print KGExploration statistics
    kgexplorer.print_statistics()

    # Save KGExploration statistics to a file
    kgexplorer.save_statistics_to_file()

    # Return statistics
    return {
        'total_prompt_tokens': kgexplorer.total_prompt_tokens,
        'total_completion_tokens': kgexplorer.total_completion_tokens,
        'total_tokens': kgexplorer.total_tokens,
        'llm_call_count': kgexplorer.llm_call_count,
        'response_times': kgexplorer.llm_response_times,
        'human_feedback_time': kgexplorer.human_feedback_time
    }


def kg_construction(universal_config):
    kgconstructor = KC.KGConstruction(**universal_config)
    all_file_list = kgconstructor.read_all_files()
    entity_type_str = kgconstructor.get_entity_type()
    relation_type_str = kgconstructor.get_relation_type()
    kgconstructor.process_entity_extraction(all_file_list, entity_type_str, kgconstructor.entity_file_path)
    text_list = []
    entity_list = []
    entity_pairs = []
    # Deduplicate entity.csv and rebuild entity pairs to avoid combinational blow-up.
    dedupe_entity_csv('../output/kg_construction/entity.csv')

    with open('../output/kg_construction/entity.csv',encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            text_list.append(row[0])
            entity_list.append(row[1])
            entity_pairs.append(row[2])
    kgconstructor.process_relation_extraction(text_list, relation_type_str, entity_list, entity_pairs, kgconstructor.relation_file_path)

    # Print KGConstruction statistics
    kgconstructor.print_statistics()

    # Save KGConstruction statistics to a file
    kgconstructor.save_statistics_to_file()

    # Return statistics
    return {
        'total_prompt_tokens': kgconstructor.total_prompt_tokens,
        'total_completion_tokens': kgconstructor.total_completion_tokens,
        'total_tokens': kgconstructor.total_tokens,
        'llm_call_count': kgconstructor.llm_call_count,
        'response_times': kgconstructor.llm_response_times
    }

def main():
    with open('../config.yaml', 'r', encoding="utf-8") as config_file:
        universal_config = yaml.safe_load(config_file)
    universal_config = resolve_runtime_config(universal_config)
    
    # Run KGExploration and collect statistics
    ke_stats = None

    # ke_stats = kg_exploration(universal_config)

    # Run KGConstruction and collect statistics (if needed)

    kc_stats = kg_construction(universal_config)


    # Print the overall statistics report
    print_total_statistics(ke_stats, kc_stats)

if __name__ == "__main__":
     main()
