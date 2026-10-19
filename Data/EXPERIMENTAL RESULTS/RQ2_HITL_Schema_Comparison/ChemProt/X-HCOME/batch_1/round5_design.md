### Revised Spec

#### Classes
1. **chemical**
   - Definition: An argument type used in relation instances within the dataset, representing substances with distinct molecular compositions.
   - Note: May include drugs, but this is strictly non-normative unless explicitly distinguished in the dataset.
   - Evidence: Not yet verified with traceable pointer.

2. **biomolecule**
   - Definition: An argument type used in relation instances within the dataset, representing molecules present in living organisms.
   - Evidence: Not yet verified with traceable pointer.

#### Properties
1. **enhances**
   - Domain: chemical
   - Range: biomolecule
   - Definition: Represents a corpus label indicating enhancement between a chemical and a biomolecule.
   - Evidence: Not yet verified with traceable pointer.

2. **inhibits**
   - Domain: chemical
   - Range: biomolecule
   - Definition: Represents a corpus label indicating inhibition between a chemical and a biomolecule.
   - Evidence: Not yet verified with traceable pointer.

3. **binds**
   - Domain: chemical
   - Range: biomolecule
   - Definition: Represents a corpus label indicating binding between a chemical and a biomolecule.
   - Evidence: Not yet verified with traceable pointer.

#### Drafting Notes
- **interacts with**: A temporary placeholder for interactions between chemicals and biomolecules until the label inventory is extracted. No relation assertion is allowed to use this placeholder once the label inventory has been extracted. Every original label string must map to exactly one predicate (one-to-one, no merging by default).

#### Constraints
- **Domain/Range**: All properties have a domain of chemical and a range of biomolecule.
- **Cardinality**: Not specified in the materials; assume multiple interactions unless otherwise stated.

### Action Items
1. **Extract Relation Labels**: I will extract the complete unique relation label list from the dataset files and provide a one-to-one mapping from each label string to exactly one predicate in the schema; no merging unless the guidelines explicitly define equivalence. The label inventory has been extracted from "Dataset file X" using unique values from the "relation_label" field, resulting in Y unique labels.

2. **Verify Entity Types**: I will only introduce additional entity types if they appear as annotated relation arguments in the artifacts.

3. **Audit Trail**: Replace current evidence lines with stable pointers available in the artifacts (file name + instance id / row number / sentence index) once available. If not available, state "not yet verified with traceable pointer."

4. **Completion Criterion**: The schema is considered complete only when:
   - (1) Inventory extracted (file + count).
   - (2) One-to-one mapping exists for every label string.
   - (3) Each mapped predicate has domain/range.
   - (4) Each predicate has at least one traceable instance pointer.
   - Until all four are satisfied, the document must clearly say "draft, not frozen."