### Revised Spec

**Entity Types:**
- Extract entity types directly from the dataset fields. The exact entity_type strings used in the records must be identified and listed. This ensures type fidelity and avoids merging distinct types under generic labels like "Organization" or "Person."

**Relationship Types:**
- Derive relationship types directly from the dataset's relationship columns. Each distinct relationship string observed in the dataset becomes a schema relation type.
- Ensure that every dataset row maps to exactly one of these types without using generic placeholders like "relatedTo."

**Directionality:**
- Validate directionality by checking real dataset rows. Ensure that the argument order follows the row structure, where the entity column is the domain and the target column is the range.
- Document the decision log to confirm that directionality is data-driven rather than based on semantic intuition.

**Evidence Traceability:**
- Define a concrete requirement for evidence traceability. Every mapped triple must retain a pointer to its provenance fields, such as source file and chunk identifiers, and at least one text span field.
- Use the existing chunk/page identifiers provided in the dataset as mandatory metadata for reproducibility.

**Action Plan:**
1. **Extract Entity Types:** Identify and list the exact entity_type strings from the dataset fields.
2. **Extract Relationship Types:** Identify and list all distinct relationship strings from the dataset's relationship columns.
3. **Validate Directionality:** Check dataset rows to ensure the correct domain and range for each relationship type.
4. **Implement Evidence Traceability:** Ensure each mapped assertion includes provenance metadata from the dataset.

**Pass/Fail Checks:**
- The schema is not eligible to freeze unless:
  - Entity types are extracted from the dataset fields.
  - Every row's relationship value maps to exactly one schema relation type without fallback placeholders.
  - Domain and range are bound to artifact-derived entity types.
  - Traceability metadata is attached to each mapped assertion.

**Clarification Questions (if necessary):**
1. Can you confirm the specific dataset fields that contain the entity_type and relationship columns?
2. Are there any additional provenance fields in the dataset that should be included for evidence traceability?
3. Is there a specific format or structure required for the traceability metadata?