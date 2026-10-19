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
1. **interactsWith** (Temporary Placeholder)
   - Domain: chemical
   - Range: biomolecule
   - Definition: A placeholder for interactions between chemicals and biomolecules until the label inventory is extracted.
   - Note: No relation assertion is allowed to use this placeholder once the label inventory has been extracted. Every original label string must map to exactly one predicate (one-to-one, no merging by default).

2. **enhances** (Pending Verification)
   - Domain: chemical
   - Range: biomolecule
   - Definition: Represents a corpus label indicating enhancement between a chemical and a biomolecule.
   - Evidence: Not yet verified with traceable pointer.

3. **inhibits** (Pending Verification)
   - Domain: chemical
   - Range: biomolecule
   - Definition: Represents a corpus label indicating inhibition between a chemical and a biomolecule.
   - Evidence: Not yet verified with traceable pointer.

4. **binds** (Pending Verification)
   - Domain: chemical
   - Range: biomolecule
   - Definition: Represents a corpus label indicating binding between a chemical and a biomolecule.
   - Evidence: Not yet verified with traceable pointer.

#### Constraints
- **Domain/Range**: All properties have a domain of chemical and a range of biomolecule.
- **Cardinality**: Not specified in the materials; assume multiple interactions unless otherwise stated.

### Action Items
1. **Extract Relation Labels**: I will extract the complete unique relation label list from the dataset files and provide a one-to-one mapping from each label string to exactly one predicate in the schema; no merging unless the guidelines explicitly define equivalence.

2. **Verify Entity Types**: I will only introduce additional entity types if they appear as annotated relation arguments in the artifacts.

3. **Audit Trail**: Replace current evidence lines with stable pointers available in the artifacts (file name + instance id / row number / sentence index) once available. If not available, state "not yet verified with traceable pointer."

4. **Label Inventory Extraction**: The label inventory has been extracted, with a count of unique labels and the exact artifact source used for extraction (e.g., "Dataset file X, containing Y unique labels"). Schema is considered complete only when every label string observed in the artifacts is mapped to one predicate and each predicate has domain/range plus at least one traceable instance pointer.