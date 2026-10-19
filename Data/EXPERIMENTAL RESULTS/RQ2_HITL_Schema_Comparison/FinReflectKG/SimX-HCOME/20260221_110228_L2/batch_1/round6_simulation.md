### Revised Spec

**Entity Types:**
- **Extraction Source:** Entity types are extracted directly from the dataset fields `entity_type` and `target_type`.
- **Inventory:** The schema's entity types are the exact distinct strings observed in these fields. No additional type strings are allowed in the schema layer.

**Relationship Types:**
- **Extraction Source:** Relationship types are derived directly from the dataset's `relationship` field.
- **Inventory:** The schema relations are the exact distinct strings observed in this field, with a one-to-one mapping and no manual consolidation.

**Directionality:**
- **Operationalization:** Domain and range are determined strictly by the row structure:
  - Domain is bound to the extracted `entity_type` value for the entity side.
  - Range is bound to the extracted `target_type` value for the target side.
  - The relationship type is the extracted relationship string.
- **Validation:** Mapping never flips arguments based on semantics; it follows the row structure consistently.

**Evidence Traceability:**
- **Provenance Bundle:** Every mapped triple retains all dataset-provided provenance columns unchanged. If renaming is needed for convenience, the original column names must be kept alongside the normalized names.

**Decision Log:**
- **Entity Type Extraction:** Entity types extracted from `entity_type` and `target_type` fields.
- **Relationship Type Extraction:** Relationship types extracted from the `relationship` field.
- **Mapping Materialization:** Mapping is materialized in the schema artifact with one-to-one correspondence to the dataset fields.

**Acceptance Gate:**
- The schema cannot freeze unless:
  1. Entity type inventory is extracted from `entity_type` and `target_type` fields.
  2. Relationship inventory is extracted from the `relationship` field.
  3. Mapping covers every distinct relationship string observed in the `relationship` field and every distinct entity type string observed in `entity_type` and `target_type`.
  4. Zero fallback to generic placeholders and zero introduction of human categories into the schema layer.
  5. Every triple retains the dataset's provenance fields without loss.

**Execution Proof:**
- The tool must demonstrate that the schema artifact was generated from extracted inventories and that the acceptance gate was run and passed. If any unseen value appears during a rerun, inventories and the schema must be regenerated.