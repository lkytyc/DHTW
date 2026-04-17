### Revised Spec

#### Classes
- **Entity Types**: Directly extracted from the dataset's `entity_type` field. Each unique value in this field is a class in the ontology.
  - **Execution Check**: Confirmed extraction from the `entity_type` field. If the field is missing, the run is marked as draft and stops.

#### Properties
- **Relation Types**: Directly extracted from the dataset's `relationship` field. Each unique value in this field is a property in the ontology.
  - **Execution Check**: Confirmed extraction from the `relationship` field. If the field is missing, the run is marked as draft and stops.

#### Constraints
- **Domain and Range**: Assigned using the dataset's row structure:
  - **Domain**: Uses the `entity_type` field.
  - **Range**: Uses the `target_type` field.
  - **Execution Check**: Confirmed domain and range assignment using extracted type strings. If fields are missing, the run is marked as draft and stops.

#### Modular Structure
- **Organizational Modules**: Modules are derived by tagging types/relations after extraction. No type/relation is excluded by module membership.
  - **Core Module**: Contains all entity types and relations derived from the dataset.
  - **Financial Module**: Organizes financial-related entity types and relations.
  - **Event Module**: Organizes event-related entity types and relations.

#### Guardrails
- **Disjoint Vocabularies**: Ensure no overlap between entity types and relation types. If overlap is detected, the tool stops and reports the overlapping strings for manual review.
- **Traceability**: Every mapped triple retains the dataset's provenance fields unchanged. If provenance fields are missing, the run is marked as draft and stops.
- **Anti-Collapse**: No fallback types such as "entity type", "thing", "other", or fallback relations such as "related to" are allowed unless those exact strings are observed in the dataset.

#### Acceptance Criteria
- **Pass/Fail Assertions**:
  - Inventories were extracted from named fields (`entity_type`, `target_type`, `relationship`).
  - Schema contains the full extracted inventories (not a hand-picked subset).
  - No generic placeholder types/relations were introduced.
  - Domain/Range are bound via row structure using extracted type strings.
  - Provenance fields are preserved for every assertion.

This revised spec ensures that the ontology is strictly grounded in the dataset's actual fields and values, with robust checks to prevent assumptions or fabrications.