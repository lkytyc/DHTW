### Revised Spec

**Decision Log:**
- Located relation label field and argument-type fields in the provided materials. (Assumption: These fields are present in the materials but specific file/field names are unknown; need access to confirm.)
- Extracted unique label inventory from the relation annotation field. (Assumption: Specific labels are unknown; need access to the relation annotation field in the artifacts to list them.)

**Candidate Classes:**
- ChemicalEntity (if only one chemical-side argument category is present)
- BioEntity (if only one bio-side argument category is present)

**Properties:**
- Generate schema relation entries directly from the extracted label inventory. (Specific labels are unknown; need access to the relation annotation field in the artifacts to list them.)

**Constraints:**
1. **Domain/Range:**
   - Each relation type must have a domain and range bound to the locked entity types (ChemicalEntity, BioEntity).
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

**Next Steps:**
1. Locate the relation label field and argument-type fields in the artifacts; record the file/field names.
2. Extract the unique label inventory and generate one schema relation entry per label.
3. Lock entity types to the argument categories observed in relation instances; remove any unused candidate classes.
4. Attach domain/range to every generated relation using argument ordering.
5. Attach one traceable instance pointer per generated relation using the standardized pointer format.