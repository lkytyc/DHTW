### Revised Spec

#### Classes
1. **chemical**
   - Definition: Represents chemical compounds, including drugs and other substances. This class encompasses all chemicals without distinguishing drugs unless explicitly stated in the dataset.

2. **gene or protein**
   - Definition: Represents genes and proteins as a merged class, unless the dataset explicitly separates them.

#### Properties
- **Global Note:** Each property represents a label used when a chemical is annotated in relation to a gene or protein, as asserted by the corpus.

1. **agonist**
   - Domain: chemical
   - Range: gene or protein
   - Definition: Label used when a chemical is annotated as an agonist of a gene or protein.

2. **antagonist**
   - Domain: chemical
   - Range: gene or protein
   - Definition: Label used when a chemical is annotated as an antagonist of a gene or protein.

3. **activator**
   - Domain: chemical
   - Range: gene or protein
   - Definition: Label used when a chemical is annotated as an activator of a gene or protein.

4. **inhibitor**
   - Domain: chemical
   - Range: gene or protein
   - Definition: Label used when a chemical is annotated as an inhibitor of a gene or protein.

5. **direct regulator**
   - Domain: chemical
   - Range: gene or protein
   - Definition: Label used when a chemical is annotated as a direct regulator of a gene or protein.

6. **indirect upregulator**
   - Domain: chemical
   - Range: gene or protein
   - Definition: Label used when a chemical is annotated as an indirect upregulator of a gene or protein.

7. **indirect downregulator**
   - Domain: chemical
   - Range: gene or protein
   - Definition: Label used when a chemical is annotated as an indirect downregulator of a gene or protein.

8. **substrate**
   - Domain: gene or protein
   - Range: chemical
   - Definition: Label used when a gene or protein is annotated as using a chemical as a substrate.

9. **produces**
   - Domain: gene or protein
   - Range: chemical
   - Definition: Label used when a gene or protein is annotated as producing a chemical.
   - Note: If the dataset uses "product of" as the label, "produces" is the normalized label used here.

#### Constraints
- **Disjoint Classes:** chemical and gene or protein are disjoint.
- **Domain and Range:** Each property has a specified domain and range as listed above.
- **Directionality:** Canonical direction is from chemical to gene or protein, except for substrate and produces, which are from gene or protein to chemical.
- **Provenance Note:** If available, minimal provenance fields include source document id and sentence span (character offsets), as optional metadata.

#### Coverage Check
- **agonist:** Not yet verified
- **antagonist:** Not yet verified
- **activator:** Not yet verified
- **inhibitor:** Not yet verified
- **direct regulator:** Not yet verified
- **indirect upregulator:** Not yet verified
- **indirect downregulator:** Not yet verified
- **substrate:** Not yet verified
- **produces:** Not yet verified

#### Supported in This Revision
- The current revision supports the nine relations listed above. If additional relations such as "part of" or "product of" are present in the dataset, they will be addressed in future revisions.

#### Deferred Relations
- Any additional relations not covered in this revision will be documented and addressed in subsequent updates.

#### Assumptions
- **Gene vs Protein:** Assumption: Gene and protein are merged into a single class unless explicitly separated in the dataset.
- **Drug Subtype:** Assumption: All chemicals are treated uniformly unless the dataset explicitly distinguishes drugs.
- **Provenance Information:** Assumption: Provenance is optional and included only if explicitly present in the dataset.

This spec is designed to be implemented directly without further clarification questions, ensuring alignment with the dataset.