### Revised Spec

**Decision Log:**
1. **Artifact Handles:**
   - **Relation Label Field:** Not yet found. Need to run a deterministic search over the provided materials to identify the exact file and field names.
   - **Argument1 Type Field:** Not yet found. Need to run a deterministic search over the provided materials to identify the exact file and field names.
   - **Argument2 Type Field:** Not yet found. Need to run a deterministic search over the provided materials to identify the exact file and field names.

**Candidate Classes:**
- **ChemicalEntity:** Locked based on observed argument categories from the artifacts.
- **BioEntity:** Locked based on observed argument categories from the artifacts.

**Properties:**
- **Relation Types:** To be extracted from the unique label inventory in the relation label field. (Specific labels are unknown; need access to the relation annotation field in the artifacts to list them.)

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
     - Decision log contains explicit artifact handles for label and argument types; no “unknown/need access” placeholders remain.
     - Schema artifact includes a non-empty relation list produced by iterating over the extracted inventory (one-to-one).
     - No relation in the schema is a generic catch-all unless it is an exact observed label string.
     - Every relation record in the schema has domain/range bound to the locked entity types based on argument ordering in the artifacts.
     - Every relation record has at least one traceable instance pointer in the standardized format.
     - If any check fails, output must be marked draft and the tool must not proceed to freeze/final export.

**Next Steps:**
1. Run deterministic search to identify file/field handles for label, argument1 type, and argument2 type; record them explicitly.
2. Extract inventory from the label field; generate relation records one-to-one into the schema artifact (non-empty).
3. Extract argument categories from the argument-type fields; lock entity types accordingly (no extra classes).
4. Assign domain/range using argument ordering; attach one pointer per relation record proving the ordering.
5. Run acceptance checks; if any fail, keep status draft and report which step failed.