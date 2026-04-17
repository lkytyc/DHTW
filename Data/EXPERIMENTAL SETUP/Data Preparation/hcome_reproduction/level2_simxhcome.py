from __future__ import annotations

import argparse
from datetime import datetime
import json
import time

from common.config import load_config
from common.io_utils import read_lines, chunk_lines
from common.project_input_loader import load_project_input
from common.llm_client import OpenAICompatibleChatClient, LLMError, TokenUsageTracker
from common.prompt_templates import (
    build_level2_simulation_prompt,
    build_level1_revision_request,
    build_level1_final_ontology_request,
)
from common.ontology_format import extract_ttl_block, ensure_ttl_has_prefixes, write_text, fix_bad_domain_list
from common.ttl_to_json_schema import convert_ttl_text_to_schema_json


def _read_multiline(prompt: str) -> tuple[str, float]:
    print(prompt)
    print("(Enter multiple lines. Type END on a new line to finish.)")
    start_time = time.time()
    lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)
    interaction_time = time.time() - start_time
    return "\n".join(lines).strip(), interaction_time


def _ask_yes_no(prompt: str) -> bool:
    while True:
        ans = input(prompt).strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("Please type 'y' or 'n' (no empty input).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    project = load_project_input(cfg.paths.project_input)
    client = OpenAICompatibleChatClient(cfg.llm)

    iter_cfg = project.get("iteration", {}) if isinstance(project.get("iteration", {}), dict) else {}
    max_rounds = int(iter_cfg.get("max_rounds", 3))

    lines = read_lines(cfg.paths.source_text, max_lines=cfg.run.max_lines)
    batches = chunk_lines(lines, cfg.run.chunk_size_lines)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_L2"
    out_dir = cfg.paths.output_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, batch in enumerate(batches, start=1):
        batch_dir = out_dir / f"batch_{i}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        # Round 1: simulated discussion
        round_idx = 1
        messages = build_level2_simulation_prompt(project, batch, cfg.run.enforce_no_hallucination)
        tracker = TokenUsageTracker()
        batch_start_time = time.time()
        total_interaction_time = 0.0

        print(f"\n[Level-2 SimX-HCOME] Round {round_idx}: Simulate KE/DE/KW discussion...")
        resp = client.chat(messages)
        tracker.update(resp)
        write_text(batch_dir / f"round{round_idx}_simulation.md", resp.content)
        messages.append({"role": "assistant", "content": resp.content})

        # Iteration rounds with supervisor interventions
        while True:
            if not cfg.run.interactive:
                print("\n(interactive=false) Skip supervisor intervention and proceed to final ontology.")
                break

            intervention, interaction_time = _read_multiline("\n--- Supervisor intervention for NEXT revision (optional):")

            if intervention:
                total_interaction_time += interaction_time
                messages.append({"role": "user", "content": "SUPERVISOR_INTERVENTION:\n" + intervention})
                messages.append({"role": "user", "content": build_level1_revision_request(cfg.run.enforce_no_hallucination)})

                round_idx += 1
                print(f"\n[Level-2 SimX-HCOME] Round {round_idx}: Revising spec/discussion based on intervention...")
                resp = client.chat(messages)
                tracker.update(resp)
                write_text(batch_dir / f"round{round_idx}_simulation.md", resp.content)
                messages.append({"role": "assistant", "content": resp.content})

            if round_idx >= max_rounds:
                print(f"\nReached max_rounds={max_rounds}. Proceeding to final ontology generation.")
                break

            cont = _ask_yes_no("Continue iteration? (y/n): ")
            if not cont:
                break

        # Final TTL generation
        print("\n[Level-2 SimX-HCOME] Final: Request final ontology (TTL)...")
        messages.append({"role": "user", "content": build_level1_final_ontology_request(project, cfg.run.enforce_no_hallucination)})
        resp2 = client.chat(messages)
        tracker.update(resp2)
        batch_total_time = time.time() - batch_start_time

        ttl_raw = extract_ttl_block(resp2.content)
        out = project["output"]
        ttl = ensure_ttl_has_prefixes(
            ttl_raw,
            out.get("namespace_prefix", "ex"),
            out.get("base_iri", "http://example.org/ontology#"),
        )

        ttl = fix_bad_domain_list(ttl)

        write_text(batch_dir / "ontology.ttl", ttl)
        write_text(batch_dir / "conversation.md", "\n\n---\n\n".join([m["content"] for m in messages if m["role"] != "system"]))

        schema = convert_ttl_text_to_schema_json(ttl, title=f"schema_batch_{i}")
        write_text(batch_dir / "schema.json", json.dumps(schema, ensure_ascii=False, indent=2))

        # Write token and time usage stats
        batch_metadata = {
            "batch_id": i,
            "rounds": round_idx,
            "llm_usage": tracker.to_dict(),
            "user_interaction_time_seconds": round(total_interaction_time, 2),
            "batch_total_time_seconds": round(batch_total_time, 2)
        }
        write_text(batch_dir / "batch_metadata.json", json.dumps(batch_metadata, ensure_ascii=False, indent=2))

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
        batch_meta_path = out_dir / f"batch_{i}" / "batch_metadata.json"
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

    meta = {
        "level": 2,
        "run_id": run_id,
        "batches": len(batches),
        "llm": {"base_url": cfg.llm.base_url, "model": cfg.llm.model},
        "iteration": {"max_rounds": max_rounds, "mode": "multiline+explicit y/n"},
    }
    write_text(out_dir / "metadata.json", json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"\nDone. Outputs written to: {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except LLMError as e:
        print(f"LLMError: {e}")
        raise
