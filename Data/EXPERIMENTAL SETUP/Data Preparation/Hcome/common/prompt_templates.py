from __future__ import annotations

from typing import Dict, Any, List


def safety_clause(enforce_no_hallucination: bool) -> str:
    if not enforce_no_hallucination:
        return ""
    return (
        "\n\nIMPORTANT SAFETY: Do not fabricate facts that are not present in the provided materials. "
        "If something is missing, explicitly say it is unknown and list what extra information is needed. "
        "Label assumptions clearly as 'Assumption:'."
    )


def build_level1_prompt_bundle(project: Dict[str, Any], source_lines: List[str], enforce_no_hallucination: bool) -> List[Dict[str, str]]:
    domain = project["domain"]
    req = project["requirements"]
    cqs = project.get("competency_questions", [])
    rules = project.get("rules_nl", [])

    system = (
        "You are an expert ontology engineer and knowledge representation specialist. "
        "You can output OWL/RDF Turtle and SWRL-like rules. "
        "You work collaboratively with a human supervisor."
        + safety_clause(enforce_no_hallucination)
    )

    user = f"""\
PROMPT 1 (generalized): Act as an ontology engineer. You will build an ontology/schema based on the data and instructions I provide.
Do NOT generate the final ontology until I explicitly ask you to output it.

PROMPT 2 (generalized 'Supervisor' stimulus): I (the human) will guide the process and may give corrections. Please be constructive.

Aim: {domain.get('aim')}
Scope: {domain.get('scope')}
Target users: {domain.get('target_users')}
Reuse policy: {domain.get('reuse_policy')}

Ontology requirements (must cover):
{chr(10).join('- ' + x for x in req.get('must_cover', []))}

Constraints:
{chr(10).join('- ' + x for x in req.get('constraints', []))}

Domain materials (one sentence per line):
{chr(10).join(source_lines)}

Competency Questions (CQs):
{chr(10).join(cqs) if cqs else '(none provided)'}

Natural language rules (optional):
{chr(10).join(rules) if rules else '(none provided)'}

Now do the following, in order:
1) Propose a candidate list of Classes and Properties (object + data properties) with brief definitions.
2) Propose a minimal modular structure (modules or top-level areas).
3) Ask me up to 5 clarification questions that would most improve correctness and completeness.
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_level1_revision_request(enforce_no_hallucination: bool) -> str:
    """
    Used for multi-round iteration: after supervisor feedback, ask the model to revise the spec again
    (still NOT final TTL).
    """
    extra = safety_clause(enforce_no_hallucination)
    return f"""\
Revise and improve the ontology design based on the supervisor feedback.

Requirements:
- Update the candidate Classes/Properties/Constraints accordingly.
- If the supervisor asks for domain/range/cardinality, incorporate them.
- Ask up to 3 clarification questions only if absolutely necessary.
- Output as a concise "Revised Spec" (NOT TTL yet).
{extra}
"""


def build_level1_final_ontology_request(project: Dict[str, Any], enforce_no_hallucination: bool) -> str:
    """
    Final request must force clean TTL output to avoid empty/invalid TTL and therefore empty schema.json.
    """
    out = project["output"]
    fmt = out.get("format", "ttl").lower()
    protege = out.get("protege_version_hint", "5.6.3")
    ns = out.get("namespace_prefix", "ex")
    base_iri = out.get("base_iri", "http://example.org/ontology#")
    extra = safety_clause(enforce_no_hallucination)

    return f"""\
PROMPT (generalized): Now develop the ontology based on ALL agreed information and corrections so far.

CRITICAL OUTPUT RULES (must follow):
- Output ONLY Turtle (TTL). No explanations, no markdown text outside the code block.
- Put the entire ontology in exactly ONE code block fenced as ```ttl ... ```.
- The code block must contain actual axioms: owl:Class / owl:ObjectProperty / owl:DatatypeProperty declarations.
- Must be openable in Protégé {protege}.
- Use prefix '{ns}' and base IRI '{base_iri}'.

Ontology content requirements:
- Include: classes, object properties, data properties.
- Add rdfs:domain and rdfs:range for each property whenever possible.
- Avoid duplicates; merge synonyms; keep names consistent.
- Translate each NL rule into SWRL-like rules if possible (or include as comments if unsure).
- Provide a short section INSIDE the TTL as comments (starting with #) named 'CQ Coverage' mapping each CQ to elements.

Output format requested: {fmt}

{extra}
"""


def build_level2_simulation_prompt(project: Dict[str, Any], source_lines: List[str], enforce_no_hallucination: bool) -> List[Dict[str, str]]:
    domain = project["domain"]
    req = project["requirements"]
    cqs = project.get("competency_questions", [])
    rules = project.get("rules_nl", [])

    system = (
        "You are a multi-persona simulation that can role-play three HCOME roles: "
        "Knowledge Engineer (KE), Domain Expert (DE), and Knowledge Worker (KW). "
        "You will simulate an iterative discussion and converge on an ontology design plan. "
        "A human supervisor may interrupt with notes; you must incorporate them."
        + safety_clause(enforce_no_hallucination)
    )

    user = f"""\
Create three personas: [KE], [DE], [KW].
Simulate a structured discussion with numbered turns. Each turn must include:
- Speaker tag ([KE]/[DE]/[KW])
- A short message (<=120 words)
- 'Decision/Action' line

Domain: {domain.get('name')}
Aim: {domain.get('aim')}
Scope: {domain.get('scope')}

Must-cover knowledge:
{chr(10).join('- ' + x for x in req.get('must_cover', []))}

Constraints:
{chr(10).join('- ' + x for x in req.get('constraints', []))}

Materials (one sentence per line):
{chr(10).join(source_lines)}

Competency Questions:
{chr(10).join(cqs) if cqs else '(none provided)'}

Natural language rules (optional):
{chr(10).join(rules) if rules else '(none provided)'}

Task:
1) Run 2 discussion rounds to identify key concepts and relations.
2) Produce a 'Draft Spec' that lists candidate classes, properties, and 3-7 key relation patterns.
3) Ask the human supervisor 3-5 specific questions to confirm.
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_level3_single_shot_prompt(project: Dict[str, Any], source_lines: List[str], enforce_no_hallucination: bool) -> List[Dict[str, str]]:
    domain = project["domain"]
    req = project["requirements"]
    cqs = project.get("competency_questions", [])
    rules = project.get("rules_nl", [])
    out = project["output"]
    fmt = out.get("format", "ttl").lower()
    protege = out.get("protege_version_hint", "5.6.3")
    ns = out.get("namespace_prefix", "ex")
    base_iri = out.get("base_iri", "http://example.org/ontology#")

    system = (
        "You are an autonomous ontology engineering simulator that will play three roles (KE, DE, KW) "
        "and produce the final ontology without human interaction."
        + safety_clause(enforce_no_hallucination)
    )

    user = f"""\
Sim-HCOME (generalized, single-shot):
Create three instances/personas of yourself: Knowledge Engineer (KE), Domain Expert (DE), Knowledge Worker (KW).
They must collaborate to create an ontology/schema.

Domain: {domain.get('name')}
Aim: {domain.get('aim')}
Scope: {domain.get('scope')}

Role responsibilities:
- KE: requirements specification + conceptualization, coherence, naming conventions.
- DE: domain terminology, definitions, constraints, edge cases.
- KW: user needs, competency questions, practical usability.

Must-cover knowledge:
{chr(10).join('- ' + x for x in req.get('must_cover', []))}

Constraints:
{chr(10).join('- ' + x for x in req.get('constraints', []))}

Materials (one sentence per line):
{chr(10).join(source_lines)}

Competency Questions:
{chr(10).join(cqs) if cqs else '(none provided)'}

Rules (NL):
{chr(10).join(rules) if rules else '(none provided)'}

CRITICAL OUTPUT RULES:
- Output ONLY Turtle (TTL) inside exactly ONE fenced code block ```ttl ... ```.
- No explanations outside the TTL block.
- Must declare owl:Class / owl:ObjectProperty / owl:DatatypeProperty.

Output:
1) Show the iterative discussion (concise) until convergence (as TTL comments only, starting with #).
2) Output the final ontology in {fmt} with prefix '{ns}' and base IRI '{base_iri}', compatible with Protégé {protege}.
3) Translate each NL rule into SWRL-like rules aligned with your classes and properties (or add as comments if unsure).
4) Provide 'CQ Coverage' as TTL comments.
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
