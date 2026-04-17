### Revised Spec

**Entity Types:**
- **Extraction Source:** Entity types are extracted directly from the dataset fields. The exact field names used in the dataset for entity types must be identified and listed. This ensures type fidelity and avoids merging distinct types under generic labels.
- **Example Entity Types:** Based on the dataset, entity types might include specific labels such as `Company`, `Executive`, `ProductLine`, `Region`, `Risk`, `Metric`, etc. These are placeholders and should be replaced with the exact strings from the dataset.

**Relationship Types:**
- **Extraction Source:** Relationship types are derived directly from the dataset's relationship columns. Each distinct relationship string observed in the dataset becomes a schema relation type.
- **No Placeholders:** No generic placeholders like "relatedTo" are allowed unless they are exact observed relationship values in the dataset. If any record contains an unseen relationship value, the inventory must be extended, and the schema regenerated.
- **Example Relationship Types:** Possible relationships might include `hasSegment`, `offersIn`, `facesRisk`, `reports`, etc., based on the dataset's actual values.

**Directionality:**
- **Operationalization:** Domain and range are determined strictly by the row structure: the entity column is always the subject side, and the target column is always the object side.
- **Validation:** This is validated by sampling real rows and recording the validation result in the decision log. No semantic overrides are allowed unless the dataset row format contradicts the assumption.

**Evidence Traceability:**
- **Mandatory Provenance Bundle:** Every mapped triple must include a mandatory provenance bundle using the dataset’s own field names. This includes:
  - `sourceFile`: The file from which the data was extracted.
  - `chunkID`: The identifier for the specific chunk or page.
  - `textSpan`: The specific text span or line number, if available.
- **Preservation:** This bundle is preserved during mapping without introducing new evidence constructs. If a triple lacks the mandatory provenance fields, it is flagged as incomplete.

**Action Plan:**
1. **Extract Entity Types:** Identify and list the exact entity_type strings from the dataset fields.
2. **Extract Relationship Types:** Identify and list all distinct relationship strings from the dataset's relationship columns.
3. **Validate Directionality:** Check dataset rows to ensure the correct domain and range for each relationship type.
4. **Implement Evidence Traceability:** Ensure each mapped assertion includes the mandatory provenance bundle.

**Pass/Fail Statement:**
- The schema cannot freeze unless it has:
  1. Artifact-derived entity types bound to domain/range.
  2. Artifact-derived relationship inventory materialized as schema relations.
  3. Zero generic placeholders unless observed as exact values.
  4. Mandatory provenance fields attached for every mapped assertion.