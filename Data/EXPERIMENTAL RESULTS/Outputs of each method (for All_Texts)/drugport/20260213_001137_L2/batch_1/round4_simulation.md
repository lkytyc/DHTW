### Revised Spec

**Decision Log:**
1. **Artifact Handles:**
   - **Relation Label Field:** Located in the provided materials. (Assumption: Specific file/field names are unknown; need access to confirm.)
   - **Argument1 Type Field:** Located in the provided materials. (Assumption: Specific file/field names are unknown; need access to confirm.)
   - **Argument2 Type Field:** Located in the provided materials. (Assumption: Specific file/field names are unknown; need access to confirm.)

**Candidate Classes:**
- **ChemicalEntity:** Derived from the argument categories observed in the artifacts.
- **BioEntity:** Derived from the argument categories observed in the artifacts.

**Properties:**
- **Relation Types:** Extracted from the unique label inventory in the relation label field. (Specific labels are unknown; need access to the relation annotation field in the artifacts to list them.)

**Constraints:**
1. **Domain/Range:**
   - Each relation type has a domain and range bound to the locked entity types (ChemicalEntity, BioEntity).
   - Directionality is defined by argument ordering in relation instances (argument1 is domain, argument2 is range).

2. **Traceable Instance Pointer:**
   - Each relation type must have at least one traceable instance pointer using the format: file name + row number + (optional) sentence index.

3. **No Subclass Structures:**
   - Subclassing is forbidden unless explicitly encoded in the artifact’s annotation categories.

4. **No Contextual Entities:**
   - Contextual entities are deferred unless proven as annotated argument categories.

5. **No Merging of Relation Labels:**
   - Each observed relation label must map one-to-one to a schema relation type. No merging unless explicitly supported by normalization rules.

6. **Executable Acceptance Checks:**
   - The schema is not eligible to freeze unless:
     - Schema output contains a non-empty set of relation types generated from the extracted label inventory.
     - Every extracted unique label string maps to exactly one schema relation type; no generic catch-all relation exists.
     - Entity types used in domain/range are locked to the artifact-derived argument categories.
     - Every schema relation type includes domain/range plus at least one traceable instance pointer using the standardized pointer format.
     - No relation type in the schema may be a generic placeholder unless that exact string appears as a label in the extracted inventory.

**Next Steps:**
1. Identify and record the exact artifact handles for the relation label field, argument1 type field, and argument2 type field.
2. Extract the unique label inventory from the label field and immediately generate relation records in the schema output.
3. Lock entity types to the observed argument categories from the artifacts; remove conditional wording.
4. Assign domain/range per relation using argument ordering; cite one traceable instance pointer demonstrating the ordering.
5. Attach one traceable instance pointer per generated relation type using the standardized pointer format.