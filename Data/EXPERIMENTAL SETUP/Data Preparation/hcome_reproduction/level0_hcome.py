from __future__ import annotations
import argparse, json
from datetime import datetime
from common.config import load_config
from common.io_utils import read_lines, chunk_lines
from common.project_input_loader import load_project_input
from common.level0_builder import interactive_level0
from common.ontology_format import write_text
import json
from common.ttl_to_json_schema import convert_ttl_text_to_schema_json

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    project = load_project_input(cfg.paths.project_input)
    lines = read_lines(cfg.paths.source_text, max_lines=cfg.run.max_lines)
    batches = chunk_lines(lines, cfg.run.chunk_size_lines)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_L0"
    out_dir = cfg.paths.output_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, batch in enumerate(batches, start=1):
        ttl = interactive_level0(batch, project)
        write_text(out_dir / f"ontology_{i}.ttl", ttl)

    schema = convert_ttl_text_to_schema_json(ttl, title=f"schema_{i}")
    write_text(out_dir / f"schema_{i}.json", json.dumps(schema, ensure_ascii=False, indent=2))

    write_text(out_dir / "metadata.json", json.dumps({"level":0,"run_id":run_id,"batches":len(batches)}, indent=2))
    print(f"\nDone. Outputs written to: {out_dir}")

if __name__ == "__main__":
    main()
