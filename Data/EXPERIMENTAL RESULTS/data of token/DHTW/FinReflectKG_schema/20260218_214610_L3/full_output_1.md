```ttl
# Iterative Discussion:
# Step 1: Identify core entity types from the materials.
# Step 2: Determine type-to-type relations explicitly mentioned.
# Step 3: Establish hierarchy and directional relations.
# Step 4: Address underspecified relations and ensure evidence traceability.
# Step 5: Finalize ontology with SWRL-like rules and CQ coverage.

@prefix ex: <http://example.org/financial-disclosure-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

# Classes
ex:Entity a owl:Class .
ex:Organization a owl:Class ; rdfs:subClassOf ex:Entity .
ex:Person a owl:Class ; rdfs:subClassOf ex:Entity .
ex:BusinessSegment a owl:Class ; rdfs:subClassOf ex:Entity .
ex:ProductOrService a owl:Class ; rdfs:subClassOf ex:Entity .
ex:MarketOrRegion a owl:Class ; rdfs:subClassOf ex:Entity .
ex:RiskFactor a owl:Class ; rdfs:subClassOf ex:Entity .
ex:FinancialMetric a owl:Class ; rdfs:subClassOf ex:Entity .
ex:Event a owl:Class ; rdfs:subClassOf ex:Entity .
ex:RegulatoryRequirement a owl:Class ; rdfs:subClassOf ex:Entity .

# Object Properties
ex:operatesIn a owl:ObjectProperty ;
    rdfs:domain ex:Organization ;
    rdfs:range ex:MarketOrRegion .

ex:produces a owl:ObjectProperty ;
    rdfs:domain ex:Organization ;
    rdfs:range ex:ProductOrService .

ex:suppliesTo a owl:ObjectProperty ;
    rdfs:domain ex:Organization ;
    rdfs:range ex:Organization .

ex:partnersWith a owl:ObjectProperty ;
    rdfs:domain ex:Organization ;
    rdfs:range ex:Organization .

ex:competesWith a owl:ObjectProperty ;
    rdfs:domain ex:Organization ;
    rdfs:range ex:Organization .

ex:hasSegment a owl:ObjectProperty ;
    rdfs:domain ex:Organization ;
    rdfs:range ex:BusinessSegment .

ex:hasRiskFactor a owl:ObjectProperty ;
    rdfs:domain ex:Organization ;
    rdfs:range ex:RiskFactor .

ex:hasMetric a owl:ObjectProperty ;
    rdfs:domain ex:Organization ;
    rdfs:range ex:FinancialMetric .

ex:hasEvent a owl:ObjectProperty ;
    rdfs:domain ex:Organization ;
    rdfs:range ex:Event .

ex:hasRegulatoryRequirement a owl:ObjectProperty ;
    rdfs:domain ex:Organization ;
    rdfs:range ex:RegulatoryRequirement .

# Hierarchy
ex:DisclosureItem a owl:Class .
ex:RiskFactor rdfs:subClassOf ex:DisclosureItem .

# SWRL-like Rules (as comments)
# Rule 1: If an Organization operatesIn a MarketOrRegion, it may have a BusinessSegment in that region.
# Rule 2: If an Organization produces a ProductOrService, it may supplyTo another Organization.
# Rule 3: If an Organization hasRiskFactor, it may affect its FinancialMetric.

# CQ Coverage:
# - What are the main entity types mentioned in the disclosures?
# - What type-to-type relations are stated between the reporting organization and other entities?
# - Which relations are directional in the materials, and what is the consistent direction choice for each?
# - Do the materials distinguish internal structures versus external actors, and how should those be represented?
# - What relations connect risks, events, and outcomes, and what should be excluded as speculation?
# - Which relations are supported by tables and how should they map to type-to-type relations?
# - What hierarchy is explicitly supported?
# - Which candidate relations are underspecified/ambiguous and should be omitted unless the materials are explicit?
```