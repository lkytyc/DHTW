### Candidate List of Classes and Properties

#### Classes
1. **ChemicalSubstance**
   - Definition: A substance with a distinct molecular composition, including drugs and other chemical compounds.
   
2. **Drug**
   - Definition: A type of ChemicalSubstance specifically used for therapeutic purposes.
   - Subclass of: ChemicalSubstance

3. **Biomolecule**
   - Definition: A molecule that is present in living organisms, including proteins and genes.

4. **Protein**
   - Definition: A type of Biomolecule composed of amino acids, involved in various biological functions.
   - Subclass of: Biomolecule

5. **Gene**
   - Definition: A type of Biomolecule that contains the information necessary for the synthesis of proteins.
   - Subclass of: Biomolecule

6. **Disease**
   - Definition: A disorder or condition affecting the health of an organism.

7. **Species**
   - Definition: A group of organisms sharing common characteristics and capable of interbreeding.

8. **CellLine**
   - Definition: A population of cells derived from a single cell and used for biological research.

9. **Tissue**
   - Definition: An ensemble of similar cells and their extracellular matrix from the same origin that together carry out a specific function.

10. **PathwayProcess**
    - Definition: A series of actions or steps taken in order to achieve a particular end in biological systems.

#### Properties
1. **interactsWith**
   - Domain: ChemicalSubstance
   - Range: Biomolecule
   - Definition: Indicates a chemical substance interacts with a biomolecule.

2. **enhancesActivityOf**
   - Domain: ChemicalSubstance
   - Range: Protein
   - Definition: Indicates a chemical substance enhances the activity of a protein.

3. **inhibitsActivityOf**
   - Domain: ChemicalSubstance
   - Range: Protein
   - Definition: Indicates a chemical substance inhibits the activity of a protein.

4. **bindsTo**
   - Domain: ChemicalSubstance
   - Range: Protein
   - Definition: Indicates a chemical substance binds to a protein.

5. **expressedIn**
   - Domain: Biomolecule
   - Range: Tissue
   - Definition: Indicates a biomolecule is expressed in a specific tissue.

6. **associatedWith**
   - Domain: Biomolecule
   - Range: Disease
   - Definition: Indicates a biomolecule is associated with a disease.

7. **foundIn**
   - Domain: Biomolecule
   - Range: Species
   - Definition: Indicates a biomolecule is found in a specific species.

8. **derivedFrom**
   - Domain: CellLine
   - Range: Tissue
   - Definition: Indicates a cell line is derived from a specific tissue.

### Minimal Modular Structure

1. **Chemical Module**
   - Classes: ChemicalSubstance, Drug
   - Properties: interactsWith, enhancesActivityOf, inhibitsActivityOf, bindsTo

2. **Biomolecule Module**
   - Classes: Biomolecule, Protein, Gene
   - Properties: expressedIn, associatedWith, foundIn

3. **Contextual Module**
   - Classes: Disease, Species, CellLine, Tissue, PathwayProcess
   - Properties: derivedFrom

### Clarification Questions

1. **Directionality of Relations:** Are there any specific relations in the materials that are directional, and how should this be represented in the schema?

2. **Interaction Categories:** Do the materials distinguish different interaction categories (e.g., binding vs regulation vs metabolism), and should the schema keep them separate or merge them?

3. **Contextual Entities:** Are there any additional contextual entities (e.g., Disease, Species, CellLine, Tissue, Pathway/Process) that must be included as types based on the materials?

4. **Relation Ambiguity:** Are there any relation candidates that are underspecified or ambiguous in the materials and should be excluded unless explicitly stated?

5. **Subclass Structure:** Is there any subclass structure explicitly supported (e.g., Drug ⊆ Chemical, Protein ⊆ Biomolecule), and what should it be?