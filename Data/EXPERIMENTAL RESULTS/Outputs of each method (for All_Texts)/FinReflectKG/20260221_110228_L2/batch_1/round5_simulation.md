### Revised Spec

**Entity Types:**
- **Extraction Source:** Entity types are extracted directly from the dataset fields. The exact field names used for entity types are `entity_type` and `target_type`.
- **Inventory:** The schema's entity types are the exact distinct strings observed in these fields. This ensures type fidelity and prevents merging into generic categories.

**Relationship Types:**
- **Extraction Source:** Relationship types are derived directly from the dataset's `relationship` field.
- **Inventory:** The schema relations are the exact distinct strings observed in this field, with a one-to-one mapping and no manual consolidation.

**Directionality:**
- **Operationalization:** Domain and range are determined strictly by the row structure:
  - Domain is bound to the extracted `entity_type` value for the entity side.
  - Range is bound to the extracted `target_type` value for the target side.
  - The relationship type is the extracted relationship string.
- **Validation:** This is validated by sampling real rows and recording the validation result in the decision log.

**Evidence Traceability:**
- **Provenance Bundle:** Every mapped triple retains all dataset-provided traceability fields unchanged. This includes fields such as `source_file`, `chunk_id`, and any other relevant fields provided in the dataset.
- **Preservation:** Any normalization must be reversible, and no original columns are dropped.

**Decision Log:**
- **Entity Type Extraction:** Entity types extracted from `entity_type` and `target_type` fields.
- **Relationship Type Extraction:** Relationship types extracted from `relationship` field.
- **Mapping Materialization:** Mapping is materialized in the schema artifact with one-to-one correspondence to the dataset fields.

**Pass/Fail Statement:**
- The schema cannot freeze unless:
  1. Entity type inventory is extracted from `entity_type` and `target_type` fields.
  2. Relationship inventory is extracted from the `relationship` field.
  3. Mapping is materialized in the schema artifact with no fallback to generic buckets.
  4. Every triple retains the dataset's provenance fields without loss.

**Acceptance Gate:**
- If any record contains an `entity_type`, `target_type`, or `relationship` value not represented in the schema inventories, the output remains draft, and the inventories must be regenerated. No fallback mapping to generic buckets is permitted.