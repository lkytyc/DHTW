**Revised Spec**

**Classes:**
1. **ChemicalSubstance**: Represents any chemical entity, including drugs and ions.
   - Evidence: "Mn(2+) ion" in sentence 1.
2. **Drug**: A specific type of ChemicalSubstance used for therapeutic purposes.
   - Evidence: "cisapride" in sentence 4.
3. **Protein**: Represents proteins involved in interactions.
   - Evidence: "beta-1,4-galactosyltransferase I" in sentence 1.
4. **Gene**: Represents genes involved in interactions.
   - Evidence: "Delta6-desaturase gene" in sentence 2.

**Properties:**
1. **enhanceActivity**: Represents the enhancement of activity by one entity on another.
   - Domain: ChemicalSubstance, Drug
   - Range: Protein
   - Evidence: "alpha-lactalbumin (LA) enhances this Glc-T activity" in sentence 1.
2. **inhibitExpression**: Represents the inhibition of expression by one entity on another.
   - Domain: ChemicalSubstance, Drug
   - Range: Gene
   - Evidence: "Emodin-6-O-β-D-glucoside inhibits HMGB1-induced inflammatory responses" in sentence 5.
3. **bindTo**: Represents binding interactions between entities.
   - Domain: Protein
   - Range: ChemicalSubstance
   - Evidence: "UDP-Gal- and UDP-Glc-bound beta4Gal-T1" in sentence 1.
4. **regulateExpression**: Represents regulation of expression by one entity on another.
   - Domain: Gene
   - Range: Protein
   - Evidence: "FADS2 affects the PUFA biosynthetic pathway" in sentence 2.

**Constraints:**
- **Directionality**: All relations are directional, with the domain acting on the range.
- **Subclassing**: No subclassing unless explicitly encoded in the annotation layer.
- **Cardinality**: Each relation connects one domain instance to one range instance.

**Decision Log:**
1. **ChemicalSubstance Class**: Derived from explicit mention of chemical entities like ions and drugs. Evidence from sentence 1.
2. **Drug Class**: Specific mention of therapeutic chemicals. Evidence from sentence 4.
3. **Protein Class**: Mention of proteins involved in interactions. Evidence from sentence 1.
4. **Gene Class**: Mention of genes involved in interactions. Evidence from sentence 2.
5. **enhanceActivity Property**: Derived from explicit enhancement interactions. Evidence from sentence 1.
6. **inhibitExpression Property**: Derived from explicit inhibition interactions. Evidence from sentence 5.
7. **bindTo Property**: Derived from explicit binding interactions. Evidence from sentence 1.
8. **regulateExpression Property**: Derived from explicit regulation interactions. Evidence from sentence 2.

**Freeze Criteria:**
- Ensure every observed label is mapped one-to-one.
- Ensure every mapped relation has domain/range plus at least one traceable instance pointer.

**Clarification Questions:**
1. Are there any additional entity types or relations explicitly mentioned in the materials that we might have missed?
2. Should we include any contextual entities as types if they are relevant to the interactions?
3. Is there any subclass structure explicitly supported by the materials that we should incorporate?