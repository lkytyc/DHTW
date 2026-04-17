```ttl
# Iterative Discussion:
# KE: We need to identify core entity types and relations. From the materials, we have Chemicals/Drugs, Proteins/Genes, and interactions like binding, regulation, and metabolism.
# DE: Agreed. We should also consider contextual entities like Disease, Species, CellLine, Tissue, and Pathway/Process, as they appear in the materials.
# KW: For practical usability, let's ensure we cover competency questions about entity types, relations, and directionality.
# KE: Directionality is important. We should represent it consistently, perhaps using 'hasInteractionWith' and 'isInteractedBy'.
# DE: The materials mention specific interactions like enhancement and inhibition, which should be separate relations.
# KW: Let's keep the schema compact but ensure it covers the main interaction types and contextual entities.
# KE: We should define a hierarchy where Drug is a subclass of Chemical, and Protein is a subclass of Biomolecule.
# DE: Agreed. This aligns with the materials and helps in deduplication.
# KW: Let's finalize the ontology with these considerations.

@prefix ex: <http://example.org/biomed-interaction-schema#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .

ex:Chemical a owl:Class .
ex:Drug a owl:Class ;
    rdfs:subClassOf ex:Chemical .

ex:Biomolecule a owl:Class .
ex:Protein a owl:Class ;
    rdfs:subClassOf ex:Biomolecule .
ex:Gene a owl:Class ;
    rdfs:subClassOf ex:Biomolecule .

ex:Disease a owl:Class .
ex:Species a owl:Class .
ex:CellLine a owl:Class .
ex:Tissue a owl:Class .
ex:Pathway a owl:Class .

ex:hasInteractionWith a owl:ObjectProperty ;
    rdfs:domain ex:Chemical ;
    rdfs:range ex:Biomolecule .

ex:isInteractedBy a owl:ObjectProperty ;
    rdfs:domain ex:Biomolecule ;
    rdfs:range ex:Chemical .

ex:enhances a owl:ObjectProperty ;
    rdfs:domain ex:Chemical ;
    rdfs:range ex:Biomolecule .

ex:inhibits a owl:ObjectProperty ;
    rdfs:domain ex:Chemical ;
    rdfs:range ex:Biomolecule .

ex:bindsTo a owl:ObjectProperty ;
    rdfs:domain ex:Chemical ;
    rdfs:range ex:Biomolecule .

ex:regulates a owl:ObjectProperty ;
    rdfs:domain ex:Chemical ;
    rdfs:range ex:Biomolecule .

ex:metabolizes a owl:ObjectProperty ;
    rdfs:domain ex:Chemical ;
    rdfs:range ex:Biomolecule .

# SWRL-like Rules (as comments):
# Rule 1: If a Chemical hasInteractionWith a Biomolecule, then the Biomolecule isInteractedBy the Chemical.
# Rule 2: If a Drug enhances a Protein, then the Protein isInteractedBy the Drug.
# Rule 3: If a Chemical inhibits a Gene, then the Gene isInteractedBy the Chemical.

# CQ Coverage:
# - Core entity types: Chemical, Drug, Biomolecule, Protein, Gene, Disease, Species, CellLine, Tissue, Pathway.
# - Type-to-type relations: hasInteractionWith, isInteractedBy, enhances, inhibits, bindsTo, regulates, metabolizes.
# - Directionality: Represented by hasInteractionWith and isInteractedBy.
# - Interaction categories: Separate relations for binding, regulation, metabolism, enhancement, inhibition.
# - Contextual entities: Included as types.
# - Subclass structure: Drug ⊆ Chemical, Protein ⊆ Biomolecule.
```