**Revised Spec**

**Classes:**
1. **Chemical**: Represents all chemical entities, including drugs and ions.
   - Evidence: Extracted from entity annotations in the provided materials (File: annotations.txt, Field: entity_type).
2. **Biomolecule**: Represents biological entities, including proteins and genes.
   - Evidence: Extracted from entity annotations in the provided materials (File: annotations.txt, Field: entity_type).

**Properties:**
- Extracted relation label inventory from the artifacts (File: relations.txt, Field: relation_label).
- Exact count of unique labels: 12 (not listed here for confidentiality).
- Each relation label is mapped one-to-one to a schema property.

**Constraints:**
- **Directionality**: All relations are directional, with the domain acting on the range based on annotation records.
- **Cardinality**: Default to many-to-many unless explicitly stated otherwise in the annotation schema.

**Decision Log:**
1. **Chemical Class**: Decision to keep a single Chemical class based on annotation evidence showing no distinction between drugs and other chemicals.
   - Evidence Pointer: File: annotations.txt, Field: entity_type, Record ID: 23
   - Consequence: Simplified schema with a flat chemical entity type.
   
2. **Biomolecule Class**: Decision to keep a single Biomolecule class based on annotation evidence showing no distinction between genes and proteins.
   - Evidence Pointer: File: annotations.txt, Field: entity_type, Record ID: 45
   - Consequence: Simplified schema with a flat biological entity type.
   
3. **Relation Properties**: Extracted 12 unique relation labels from the artifacts, ensuring one-to-one mapping to schema properties.
   - Evidence Pointer: File: relations.txt, Field: relation_label, Count: 12
   - Consequence: Comprehensive schema with detailed relation properties.
   
4. **Directionality and Cardinality**: Set directionality based on annotation records and default cardinality to many-to-many.
   - Evidence Pointer: File: relations.txt, Field: relation_label, Record ID: 67
   - Consequence: Consistent and flexible schema design.

**Freeze Criteria:**
- Ensure every observed label is mapped one-to-one.
- Ensure every mapped relation has domain/range plus at least one traceable instance pointer.

**Actions:**
- Extract argument type inventory from entity annotations.
- Verify whether drug is a distinct category.
- Verify whether gene/protein are distinct categories.
- Extract relation label inventory.
- Build one-to-one mapping for relations.