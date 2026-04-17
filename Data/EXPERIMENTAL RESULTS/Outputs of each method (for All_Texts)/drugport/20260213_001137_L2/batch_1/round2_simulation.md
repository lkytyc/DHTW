### Revised Spec

**Candidate Classes:**
- Chemical
- Drug
- Protein
- Gene
- Biomolecule

**Properties:**
- Extract relation labels directly from the provided materials. (Specific labels are unknown; need access to the relation annotation field in the artifacts to list them.)

**Constraints:**
1. **Domain/Range:** 
   - Each relation type must have a domain and range bound to the fixed entity types (Chemical, Drug, Protein, Gene, Biomolecule).
   - Directionality is defined by argument ordering in relation instances (argument1 is domain, argument2 is range).

2. **Traceable Instance Pointer:**
   - Each relation type must have at least one traceable instance pointer (file + record handle) proving it exists in the artifacts.

3. **No Subclass Structures:**
   - Subclassing is forbidden unless explicitly encoded in the artifact’s annotation categories.

4. **No Contextual Entities:**
   - Contextual entities (Disease, Species, CellLine, Tissue, Pathway/Process) are deferred unless proven as annotated argument categories.

5. **No Merging of Relation Labels:**
   - Each observed relation label must map one-to-one to a schema relation type. No merging unless explicitly supported by normalization rules.

6. **Executable Acceptance Checks:**
   - The schema is not eligible to freeze unless:
     - Every distinct relation label observed in the artifacts has exactly one mapped schema relation type.
     - The schema relation set contains no generic catch-all placeholders.
     - Every relation type has domain/range bound to the fixed entity types.
     - Every relation type has at least one traceable instance pointer proving it exists in the artifacts.

**Clarification Questions:**
1. Can you provide access to the specific file/field where relation labels are annotated in the artifacts?
2. Are there any specific guidelines or normalization rules available for relation label equivalence?
3. Is there a preferred format for documenting traceable instance pointers (e.g., file name, record handle)?