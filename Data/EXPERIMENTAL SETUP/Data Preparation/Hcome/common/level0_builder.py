from __future__ import annotations
from typing import List, Dict, Any
import re

def guess_terms_from_lines(lines: List[str], max_terms: int = 60) -> List[str]:
    candidates = set()
    for ln in lines:
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}", ln):
            if token.lower() in {"the","and","then","if","else","when","with","from","into","that","this"}:
                continue
            if token[0].isupper() or token.isupper():
                candidates.add(token)
    if not candidates:
        for ln in lines:
            parts = ln.split()
            if parts:
                candidates.add(parts[0])
    return list(sorted(candidates))[:max_terms]

def build_minimal_ttl(prefix: str, base_iri: str, classes: List[str], obj_props: List[str], data_props: List[str]) -> str:
    def iri(name: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_]", "_", name.strip()) or "Thing"
        if safe[0].isdigit():
            safe = "T_" + safe
        return f"{prefix}:{safe}"

    lines = [
        f"@prefix {prefix}: <{base_iri}> .",
        "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
        "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
        "",
        f"{prefix}:Ontology a owl:Ontology .",
        ""
    ]
    for c in classes:
        lines.append(f"{iri(c)} a owl:Class ; rdfs:label \"{c}\"@en .")
    lines.append("")
    for p in obj_props:
        lines.append(f"{iri(p)} a owl:ObjectProperty ; rdfs:label \"{p}\"@en .")
    lines.append("")
    for p in data_props:
        lines.append(f"{iri(p)} a owl:DatatypeProperty ; rdfs:label \"{p}\"@en .")
    lines.append("")
    return "\n".join(lines)

def interactive_level0(lines: List[str], project: Dict[str, Any]) -> str:
    out = project["output"]
    prefix = out.get("namespace_prefix", "ex")
    base_iri = out.get("base_iri", "http://example.org/ontology#")

    guessed = guess_terms_from_lines(lines)
    print("\n[Level-0 HCOME] No-LLM mode. We'll build a minimal TTL skeleton.")
    print("Guessed candidate terms from your text (edit as needed):")
    print(", ".join(guessed) if guessed else "(none)")

    def ask_list(prompt: str, default: List[str]) -> List[str]:
        print("\n" + prompt)
        print("Enter comma-separated values. Press ENTER to accept default.")
        raw = input("> ").strip()
        if not raw:
            return default
        return [x.strip() for x in raw.split(",") if x.strip()]

    classes = ask_list("Classes:", guessed[:20] if guessed else ["Entity", "Event", "Actor"])
    obj_props = ask_list("Object properties (relations):", ["relatedTo", "hasPart", "occursIn"])
    data_props = ask_list("Data properties (attributes):", ["hasValue", "hasTimestamp", "hasStatus"])

    return build_minimal_ttl(prefix, base_iri, classes, obj_props, data_props)
