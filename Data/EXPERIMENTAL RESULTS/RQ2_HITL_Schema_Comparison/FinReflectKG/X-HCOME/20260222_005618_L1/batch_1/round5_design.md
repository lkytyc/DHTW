### Revised Spec

#### Classes
- **Entity Types**: Derived from the union of distinct values in the dataset's `entity_type` and `target_type` fields. Each unique value in this union is a class in the ontology.
  - **Execution Artifact**: Record the exact input source (file/split) and column names (`entity_type`, `target_type`) used for extraction. Confirm that the inventory was materialized into the schema output. The tool fails if it finds a `target_type` value not present in the class inventory.

#### Properties
- **Relation Types**: Directly extracted from the dataset's `relationship` field. Each unique value in this field is a property in the ontology.
  - **Execution Artifact**: Record the exact input source (file/split) and column name (`relationship`) used for extraction. Confirm that the inventory was materialized into the schema output.

#### Constraints
- **Domain and Range**: Assigned using the dataset's row structure:
  - **Domain**: Uses the `entity_type` field.
  - **Range**: Uses the `target_type` field.
  - **Execution Artifact**: Record the exact input source (file/split) and column names used for domain and range assignment. Confirm that these were materialized into the schema output.

#### Modular Structure
- **Organizational Modules**: Modules are tags only and do not rename, merge, or exclude items. The canonical schema remains the raw extracted strings with optional aliases stored separately.
  - **Core Module**: Contains all entity types and relations derived from the dataset.
  - **Financial Module**: Tags financial-related entity types and relations.
  - **Event Module**: Tags event-related entity types and relations.

#### Guardrails
- **Disjoint Vocabularies**: Ensure no overlap between entity types and relation types. Apply a normalization function (e.g., case folding, whitespace trimming) for overlap checks and inventory uniqueness. If overlap is detected after normalization, the tool stops and reports the overlapping strings for manual review.
- **Traceability**: Carry through all provenance columns present in the dataset unchanged. Define a minimal mandatory subset only after confirming the exact columns in the artifact schema. If any record is missing these fields, the run is marked as draft and stops.
- **Anti-Collapse**: No fallback types such as "entity type", "thing", "other", or fallback relations such as "related to" are allowed unless those exact strings are observed in the dataset.

#### Acceptance Criteria
- **Pass/Fail Assertions**:
  - **Inventories Extraction**: Pass if inventories were extracted from named fields (`entity_type`, `target_type`, `relationship`) and materialized into the schema output.
  - **Schema Completeness**: Pass if the schema contains the full extracted inventories (not a hand-picked subset).
  - **No Generic Placeholders**: Pass if no generic placeholder types/relations were introduced.
  - **Domain/Range Binding**: Pass if domain/range are bound via row structure using extracted type strings.
  - **Provenance Preservation**: Pass if all provenance fields are preserved for every assertion.
  - **Non-Trivial Inventory**: Pass if a non-trivial relationship inventory is detected and fully materialized as relation types. If compressed into a small subset, the run fails.

- **Execution**: Each acceptance condition is executed and written into the run output. If any condition fails, the run stops without producing a final schema.

This revised spec ensures that the ontology is fully grounded in the dataset's actual fields and values, with robust checks and execution artifacts to prevent assumptions or fabrications.