### Revised Spec

#### Classes
- **Entity Types**: Directly extracted from the dataset's `entity_type` field. Each unique value in this field is a class in the ontology.
  - **Execution Artifact**: Record the exact input source (file/split) and column name (`entity_type`) used for extraction. Confirm that the inventory was materialized into the schema output.

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
- **Disjoint Vocabularies**: Ensure no overlap between entity types and relation types. If overlap is detected, the tool stops and reports the overlapping strings for manual review.
- **Traceability**: Every mapped triple retains the dataset's provenance fields unchanged. The exact provenance fields are `entity_id`, `target_id`, and `source_file`. If any record is missing these fields, the run is marked as draft and stops.
- **Anti-Collapse**: No fallback types such as "entity type", "thing", "other", or fallback relations such as "related to" are allowed unless those exact strings are observed in the dataset.

#### Acceptance Criteria
- **Pass/Fail Assertions**:
  - **Inventories Extraction**: Pass if inventories were extracted from named fields (`entity_type`, `target_type`, `relationship`) and materialized into the schema output.
  - **Schema Completeness**: Pass if the schema contains the full extracted inventories (not a hand-picked subset).
  - **No Generic Placeholders**: Pass if no generic placeholder types/relations were introduced.
  - **Domain/Range Binding**: Pass if domain/range are bound via row structure using extracted type strings.
  - **Provenance Preservation**: Pass if provenance fields (`entity_id`, `target_id`, `source_file`) are preserved for every assertion.

- **Execution**: Each acceptance condition is executed and written into the run output. If any condition fails, the run stops without producing a final schema.

This revised spec ensures that the ontology is strictly grounded in the dataset's actual fields and values, with robust checks and execution artifacts to prevent assumptions or fabrications.