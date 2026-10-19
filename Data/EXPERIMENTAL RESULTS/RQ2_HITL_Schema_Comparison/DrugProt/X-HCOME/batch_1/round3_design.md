### Revised Spec

#### Classes
1. **Chemical**
   - Definition: Represents chemical compounds, including drugs and other substances. This class encompasses all chemicals without distinguishing drugs unless explicitly stated in the dataset.

2. **GeneOrProtein**
   - Definition: Represents genes and proteins as a merged class, unless the dataset explicitly separates them.

#### Properties
1. **agonist**
   - Domain: Chemical
   - Range: GeneOrProtein
   - Definition: A labeled relation asserted by the corpus indicating a chemical that acts as an agonist to a gene or protein.

2. **antagonist**
   - Domain: Chemical
   - Range: GeneOrProtein
   - Definition: A labeled relation asserted by the corpus indicating a chemical that acts as an antagonist to a gene or protein.

3. **activator**
   - Domain: Chemical
   - Range: GeneOrProtein
   - Definition: A labeled relation asserted by the corpus indicating a chemical that acts as an activator to a gene or protein.

4. **inhibitor**
   - Domain: Chemical
   - Range: GeneOrProtein
   - Definition: A labeled relation asserted by the corpus indicating a chemical that acts as an inhibitor to a gene or protein.

5. **direct regulator**
   - Domain: Chemical
   - Range: GeneOrProtein
   - Definition: A labeled relation asserted by the corpus indicating a chemical that directly regulates a gene or protein.

6. **indirect upregulator**
   - Domain: Chemical
   - Range: GeneOrProtein
   - Definition: A labeled relation asserted by the corpus indicating a chemical that indirectly upregulates a gene or protein.

7. **indirect downregulator**
   - Domain: Chemical
   - Range: GeneOrProtein
   - Definition: A labeled relation asserted by the corpus indicating a chemical that indirectly downregulates a gene or protein.

8. **substrate**
   - Domain: GeneOrProtein
   - Range: Chemical
   - Definition: A labeled relation asserted by the corpus indicating a gene or protein that uses a chemical as a substrate.

9. **produces**
   - Domain: GeneOrProtein
   - Range: Chemical
   - Definition: A labeled relation asserted by the corpus indicating a gene or protein that produces a chemical.

#### Constraints
- **Disjoint Classes:** Chemical and GeneOrProtein are disjoint.
- **Domain and Range:** Each property has a specified domain and range as listed above.
- **Directionality:** Relations are consistently represented as Chemical → GeneOrProtein, except for substrate and produces, which are GeneOrProtein → Chemical.
- **Provenance Note:** Relation assertions are derived from text-mined or curated sentences, if such provenance is present in the dataset.

#### Coverage Check
- **agonist:** Verified by example: [Example from dataset]
- **antagonist:** Verified by example: [Example from dataset]
- **activator:** Verified by example: [Example from dataset]
- **inhibitor:** Verified by example: [Example from dataset]
- **direct regulator:** Verified by example: [Example from dataset]
- **indirect upregulator:** Verified by example: [Example from dataset]
- **indirect downregulator:** Verified by example: [Example from dataset]
- **substrate:** Verified by example: [Example from dataset]
- **produces:** Verified by example: [Example from dataset]

#### Assumptions
- **Gene vs Protein:** Assumption: Gene and protein are merged into a single class unless explicitly separated in the dataset.
- **Drug Subtype:** Assumption: All chemicals are treated uniformly unless the dataset explicitly distinguishes drugs.
- **Provenance Information:** Assumption: Provenance is optional and included only if explicitly present in the dataset.

This spec is designed to be implemented directly without further clarification questions, ensuring alignment with the dataset.