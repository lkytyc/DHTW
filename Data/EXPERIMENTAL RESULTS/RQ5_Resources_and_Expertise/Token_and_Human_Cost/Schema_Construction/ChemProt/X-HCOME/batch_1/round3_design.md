### Revised Spec

#### Classes
1. **chemical**
   - Definition: A substance with a distinct molecular composition, used in the dataset to describe interactions with biomolecules.
   - Note: May include drugs, but this is a non-normative note unless explicitly distinguished in the dataset.
   - Evidence: Not yet verified with traceable pointer.

2. **biomolecule**
   - Definition: A molecule present in living organisms, used in the dataset to describe interactions with chemicals.
   - Evidence: Not yet verified with traceable pointer.

#### Properties
1. **interactsWith** (Temporary Placeholder)
   - Domain: chemical
   - Range: biomolecule
   - Definition: A temporary placeholder for interactions between chemicals and biomolecules until the label inventory is extracted.
   - Note: Will be removed once the label inventory is extracted and distinct predicates are defined.

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

3. **Audit Trail**: Replace current evidence lines with stable pointers available in the artifacts (file name + record id / sentence id / relation instance id) once available. If not available, state "not yet verified with traceable pointer."