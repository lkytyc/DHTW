from __future__ import annotations
import argparse, json
from datetime import datetime
import time
from common.config import load_config
from common.io_utils import read_lines, chunk_lines
from common.project_input_loader import load_project_input
from common.llm_client import OpenAICompatibleChatClient, LLMError, TokenUsageTracker
from common.prompt_templates import build_level3_single_shot_prompt
from common.ontology_format import extract_ttl_block, ensure_ttl_has_prefixes, write_text
import json
from common.ttl_to_json_schema import convert_ttl_text_to_schema_json



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    project = load_project_input(cfg.paths.project_input)
    client = OpenAICompatibleChatClient(cfg.llm)

    lines = read_lines(cfg.paths.source_text, max_lines=cfg.run.max_lines)
    batches = chunk_lines(lines, cfg.run.chunk_size_lines)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_L3"
    out_dir = cfg.paths.output_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, batch in enumerate(batches, start=1):
        messages = build_level3_single_shot_prompt(project, batch, cfg.run.enforce_no_hallucination)
        tracker = TokenUsageTracker()
        batch_start_time = time.time()

        print("\n[Level-3 Sim-HCOME] Single-shot autonomous run...")
        resp = client.chat(messages)
        tracker.update(resp)
        write_text(out_dir / f"full_output_{i}.md", resp.content)

        ttl_raw = extract_ttl_block(resp.content)
        out = project["output"]
        ttl = ensure_ttl_has_prefixes(ttl_raw, out.get("namespace_prefix","ex"), out.get("base_iri","http://example.org/ontology#"))
        write_text(out_dir / f"ontology_{i}.ttl", ttl)
        schema = convert_ttl_text_to_schema_json(ttl, title=f"schema_{i}")
        write_text(out_dir / f"schema_{i}.json", json.dumps(schema, ensure_ascii=False, indent=2))

        # Write token and time usage stats
        batch_total_time = time.time() - batch_start_time
        batch_metadata = {
            "batch_id": i,
            "llm_usage": tracker.to_dict(),
            "user_interaction_time_seconds": 0.0,
            "batch_total_time_seconds": round(batch_total_time, 2)
        }
        write_text(out_dir / f"batch_{i}_metadata.json", json.dumps(batch_metadata, ensure_ascii=False, indent=2))

    # Collect and write aggregated statistics
    aggregated_stats = {
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "total_llm_time_seconds": 0.0,
        "total_user_interaction_time_seconds": 0.0,
        "total_api_calls": 0,
        "total_batches": len(batches),
        "batches_detail": []
    }

    for i in range(1, len(batches) + 1):
        batch_meta_path = out_dir / f"batch_{i}_metadata.json"
        if batch_meta_path.exists():
            with open(batch_meta_path, "r", encoding="utf-8") as f:
                batch_meta = json.load(f)
                aggregated_stats["total_prompt_tokens"] += batch_meta["llm_usage"]["total_prompt_tokens"]
                aggregated_stats["total_completion_tokens"] += batch_meta["llm_usage"]["total_completion_tokens"]
                aggregated_stats["total_tokens"] += batch_meta["llm_usage"]["total_tokens"]
                aggregated_stats["total_llm_time_seconds"] += batch_meta["llm_usage"]["total_time_seconds"]
                aggregated_stats["total_api_calls"] += batch_meta["llm_usage"]["api_calls"]
                aggregated_stats["total_user_interaction_time_seconds"] += batch_meta["user_interaction_time_seconds"]
                aggregated_stats["batches_detail"].append(batch_meta)

    aggregated_stats["total_run_time_seconds"] = round(
        aggregated_stats["total_llm_time_seconds"] + aggregated_stats["total_user_interaction_time_seconds"], 2
    )
    aggregated_stats["avg_llm_time_per_call_seconds"] = round(
        aggregated_stats["total_llm_time_seconds"] / aggregated_stats["total_api_calls"], 2
    ) if aggregated_stats["total_api_calls"] > 0 else 0.0

    write_text(out_dir / "aggregated_stats.json", json.dumps(aggregated_stats, ensure_ascii=False, indent=2))

    write_text(out_dir / "metadata.json", json.dumps({"level":3,"run_id":run_id,"batches":len(batches),"llm":{"base_url":cfg.llm.base_url,"model":cfg.llm.model}}, indent=2))
    print(f"\nDone. Outputs written to: {out_dir}")

if __name__ == "__main__":
    try:
        main()
    except LLMError as e:
        print(f"LLMError: {e}")
        raise
