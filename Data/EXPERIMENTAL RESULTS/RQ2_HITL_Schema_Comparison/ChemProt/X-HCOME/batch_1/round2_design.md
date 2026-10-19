### Revised Spec

#### Classes
1. **ChemicalSubstance**
   - Definition: A substance with a distinct molecular composition, including drugs and other chemical compounds.
   - Evidence: Mentioned in multiple sentences describing chemical interactions (e.g., "MDMA", "cisapride").

2. **Biomolecule**
   - Definition: A molecule that is present in living organisms, including proteins and genes.
   - Evidence: Mentioned in sentences describing interactions with chemicals (e.g., "serotonin transporter", "FADS2 gene").

#### Properties
1. **interactsWith**
   - Domain: ChemicalSubstance
   - Range: Biomolecule
   - Definition: Indicates a chemical substance interacts with a biomolecule.
   - Evidence: General interaction statements (e.g., "MDMA-induced reductions in SERT density").

2. **enhancesActivityOf**
   - Domain: ChemicalSubstance
   - Range: Biomolecule
   - Definition: Indicates a chemical substance enhances the activity of a biomolecule.
   - Evidence: Specific enhancement statements (e.g., "alpha-lactalbumin (LA) enhances this Glc-T activity").

3. **inhibitsActivityOf**
   - Domain: ChemicalSubstance
   - Range: Biomolecule
   - Definition: Indicates a chemical substance inhibits the activity of a biomolecule.
   - Evidence: Specific inhibition statements (e.g., "CYP2E1 inhibits CYP2B4-mediated metabolism").

4. **bindsTo**
   - Domain: ChemicalSubstance
   - Range: Biomolecule
   - Definition: Indicates a chemical substance binds to a biomolecule.
   - Evidence: Binding statements (e.g., "Cisapride was found to bind the human ether-a-go-go-related gene (HERG) potassium channel").

#### Constraints
- **Domain/Range**: All properties have a domain of ChemicalSubstance and a range of Biomolecule.
- **Cardinality**: Not specified in the materials; assume multiple interactions unless otherwise stated.

#### Audit Trail
- **ChemicalSubstance**: Verified in sentences mentioning specific chemicals (e.g., "MDMA", "cisapride").
- **Biomolecule**: Verified in sentences mentioning specific proteins/genes (e.g., "serotonin transporter", "FADS2 gene").
- **interactsWith**: Verified in general interaction statements.
- **enhancesActivityOf**: Verified in enhancement statements.
- **inhibitsActivityOf**: Verified in inhibition statements.
- **bindsTo**: Verified in binding statements.

### Clarification Questions
1. **Relation Labels**: Are there any additional relation labels explicitly mentioned in the materials that need to be included?
2. **Entity Types**: Are there any other entity types explicitly annotated as participating entities in the core relation records?
3. **Cardinality**: Is there any specific cardinality constraint mentioned in the materials that should be applied to the properties?