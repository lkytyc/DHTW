from __future__ import annotations

import re
from pathlib import Path

TTL_PREAMBLE = """@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
"""


def ensure_ttl_has_prefixes(ttl: str, prefix: str, base_iri: str) -> str:
    ttl = ttl.strip()
    if f"@prefix {prefix}:" not in ttl:
        ttl = f"@prefix {prefix}: <{base_iri}> .\n" + ttl
    if "@prefix owl:" not in ttl:
        ttl = TTL_PREAMBLE + "\n" + ttl
    return ttl + "\n"


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def extract_ttl_block(text: str) -> str:
    """
    Robust Turtle extractor:
    1) Prefer largest ```ttl``` or ```turtle``` fenced block
    2) Else pick largest fenced block
    3) Else fallback from first @prefix/PREFIX
    4) Else return raw text
    """
    blocks = re.findall(r"```(?:ttl|turtle)\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        best = max((b.strip() for b in blocks), key=len, default="")
        if best:
            return best

    blocks_any = re.findall(r"```\s*(.*?)```", text, flags=re.DOTALL)
    if blocks_any:
        best = max((b.strip() for b in blocks_any), key=len, default="")
        if best:
            return best

    m = re.search(r"(^@prefix\s+|^PREFIX\s+)", text, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        return text[m.start():].strip()

    return text.strip()


def fix_bad_domain_list(ttl: str) -> str:
    """
    Safe fixer for ONLY this invalid Turtle pattern:
      rdfs:domain [ ex:A, ex:B, ex:C ] ;
    which is illegal.

    It converts it to:
      rdfs:domain [ a owl:Class ; owl:unionOf ( ex:A ex:B ex:C ) ] ;

    IMPORTANT: It will NOT touch:
    - domains that already contain 'owl:unionOf'
    - bracket contents without commas
    - complex blank nodes that look like valid OWL structures
    """
    # Only match bracket-domain that contains at least one comma, and does NOT already contain owl:unionOf
    pattern = re.compile(
        r"rdfs:domain\s*\[\s*([^\]]*?,[^\]]*?)\s*\]\s*;",
        flags=re.IGNORECASE | re.DOTALL
    )

    def repl(m: re.Match) -> str:
        inside = m.group(1).strip()

        # If it already contains OWL structure, don't touch it
        if "owl:unionOf" in inside or "a owl:Class" in inside or "(" in inside or ")" in inside or ";" in inside:
            return m.group(0)

        parts = [p.strip() for p in inside.split(",") if p.strip()]
        tokens = []
        for p in parts:
            p = p.strip().rstrip(".;")
            if p:
                tokens.append(p)

        if not tokens:
            return m.group(0)

        union = " ".join(tokens)
        return f"rdfs:domain [ a owl:Class ; owl:unionOf ( {union} ) ] ;"

    return pattern.sub(repl, ttl)
