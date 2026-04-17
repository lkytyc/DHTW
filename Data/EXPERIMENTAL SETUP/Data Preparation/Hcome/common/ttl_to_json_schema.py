# common/ttl_to_json_schema.py
from __future__ import annotations

from typing import Dict, Any, List, Set, Optional

from rdflib import Graph, URIRef, BNode
from rdflib.namespace import RDF, RDFS, OWL
from rdflib.collection import Collection


# -----------------------------
# Helpers
# -----------------------------
def _local_name(uri: URIRef) -> str:
    s = str(uri)
    if "#" in s:
        return s.split("#")[-1]
    return s.rstrip("/").split("/")[-1]


def _ensure_prefixes(ttl_text: str) -> str:
    """Guard: ensure rdf/rdfs/owl/xsd prefixes exist to prevent rdflib parse errors."""
    prefix_block = ""
    if "@prefix rdf:" not in ttl_text:
        prefix_block += "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n"
    if "@prefix rdfs:" not in ttl_text:
        prefix_block += "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n"
    if "@prefix owl:" not in ttl_text:
        prefix_block += "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
    if "@prefix xsd:" not in ttl_text:
        prefix_block += "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n"
    if prefix_block:
        ttl_text = prefix_block + "\n" + ttl_text
    return ttl_text


def _extract_union_members(g: Graph, node: BNode) -> List[str]:
    """Extract members from: [ a owl:Class ; owl:unionOf ( ex:A ex:B ) ]"""
    members: List[str] = []
    for lst in g.objects(node, OWL.unionOf):
        try:
            col = Collection(g, lst)
            for item in col:
                if isinstance(item, URIRef):
                    nm = _local_name(item)
                    if nm:
                        members.append(nm)
        except Exception:
            continue
    return members


def _extract_domain_or_range_names(g: Graph, node) -> List[str]:
    """
    Return a list of class names for domain/range.
    Handles:
      - URIRef -> [localName]
      - BNode with owl:unionOf -> [A, B, ...]
      - otherwise -> []
    """
    if node is None:
        return []
    if isinstance(node, URIRef):
        nm = _local_name(node)
        return [nm] if nm else []
    if isinstance(node, BNode):
        members = _extract_union_members(g, node)
        return members
    return []


def _first_object(g: Graph, subj: URIRef, pred: URIRef):
    for o in g.objects(subj, pred):
        return o
    return None


# -----------------------------
# Main converter
# -----------------------------
def convert_ttl_text_to_schema_json(ttl_text: str, title: str = "Schema") -> Dict[str, Any]:
    """
    Output schema JSON that contains ONLY:
      - entity type (Classes)
      - relation (ObjectProperty) where BOTH domain & range are entity types
      - hierarchy (subClassOf)

    Datatype properties are intentionally ignored (NOT exported).
    """
    ttl_text = _ensure_prefixes(ttl_text)

    g = Graph()
    g.parse(data=ttl_text, format="turtle")

    # 1) Collect classes
    classes: Set[str] = set()
    for c in g.subjects(RDF.type, OWL.Class):
        if isinstance(c, URIRef):
            nm = _local_name(c)
            if nm:
                classes.add(nm)

    for c in g.subjects(RDF.type, RDFS.Class):
        if isinstance(c, URIRef):
            nm = _local_name(c)
            if nm:
                classes.add(nm)

    entity_type = [{"id": nm, "label": nm} for nm in sorted(classes, key=lambda x: x.lower())]

    # 2) Hierarchy: rdfs:subClassOf
    hierarchy: List[Dict[str, str]] = []
    for child, parent in g.subject_objects(RDFS.subClassOf):
        if isinstance(child, URIRef) and isinstance(parent, URIRef):
            c = _local_name(child)
            p = _local_name(parent)
            if c and p and c in classes and p in classes:
                hierarchy.append({"id": "IsA", "label": "IsA", "domain": c, "range": p})

    # 3) Relations: ONLY owl:ObjectProperty, and ONLY if domain/range are entity types
    relations: List[Dict[str, str]] = []
    seen = set()

    for prop in g.subjects(RDF.type, OWL.ObjectProperty):
        if not isinstance(prop, URIRef):
            continue
        pid = _local_name(prop)
        if not pid:
            continue

        dnode = _first_object(g, prop, RDFS.domain)
        rnode = _first_object(g, prop, RDFS.range)

        d_list = _extract_domain_or_range_names(g, dnode)
        r_list = _extract_domain_or_range_names(g, rnode)

        # Keep only class-to-class relations for domain and range.
        # For unionOf, keep members that exist in classes and choose the first
        # valid candidate as the exported domain/range.
        d_candidates = [d for d in d_list if d in classes]
        r_candidates = [r for r in r_list if r in classes]

        if not d_candidates or not r_candidates:
            # drop anything that cannot be resolved to class->class
            continue

        dom = d_candidates[0]
        rng = r_candidates[0]

        key = (pid, dom, rng)
        if key in seen:
            continue
        seen.add(key)

        relations.append({
            "id": pid,
            "label": pid,
            "domain": dom,
            "range": rng,
        })

    return {
        "title": title,
        "id": title,
        "entity type": entity_type,
        "relation": relations,
        "hierarchy": hierarchy,
    }
