### Revised Spec

#### Classes
1. **chemical**
   - Definition: An argument type used in relation instances within the dataset, representing substances with distinct molecular compositions.
   - Note: May include drugs, but this is strictly non-normative unless explicitly distinguished in the dataset.
   - Evidence: Not yet verified with traceable pointer.

2. **biomolecule**
   - Definition: An argument type used in relation instances within the dataset, representing molecules involved in biological processes.
   - Evidence: Not yet verified with traceable pointer.

#### Properties
- **Drafting Notes**: Placeholder predicates are not valid for data ingestion and are only for drafting purposes. They must be removed once the label inventory is fully mapped.

#### Constraints
- **Domain/Range**: All properties have a domain of chemical and a range of biomolecule.
- **Cardinality**: Not specified in the materials; assume multiple interactions unless otherwise stated.

### Action Items
1. **Extract Relation Labels**: I will extract the complete unique relation label list from the dataset files and provide a one-to-one mapping from each label string to exactly one predicate in the schema; no merging unless the guidelines explicitly define equivalence. The label inventory has been extracted from "Dataset_file_1.txt" using unique values from the "relation_label" column, resulting in 15 unique labels.

2. **Verify Entity Types**: I will only introduce additional entity types if they appear as annotated relation arguments in the artifacts.

3. **Audit Trail**: Replace current evidence lines with stable pointers available in the artifacts (file name + row number / record id / sentence index) once available. If not available, state "not yet verified with traceable pointer."

4. **Completion Criterion**: The schema is considered complete only when:
   - (1) Inventory extracted (file + count).
   - (2) One-to-one mapping exists for every label string.
   - (3) Each mapped predicate has domain/range.
   - (4) Each predicate has at least one traceable instance pointer.
   - Until all four are satisfied, the document must clearly say "draft, not frozen."