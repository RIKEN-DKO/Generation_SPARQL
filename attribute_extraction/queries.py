q1_label="Get all properties of all classes with their labels."
q1='''
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX orth: <http://purl.org/net/orth#>
PREFIX genex: <http://purl.org/genex#>
PREFIX obo: <http://purl.obolibrary.org/obo/>
PREFIX up: <http://purl.uniprot.org/core/>


select distinct ?class ?classLabel ?property ?propertyLabel  where {


{?class a owl:Class.} UNION {?class a rdfs:Class}
?instance a ?class.
optional {?class rdfs:label ?classLabel}
?instance ?property ?value
optional {?property rdfs:label ?propertyLabel}
filter isLiteral(?value) 
} '''

q2_label="Get all properties of a given list of concepts."
q2='''
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX orth: <http://purl.org/net/orth#>
PREFIX genex: <http://purl.org/genex#>
PREFIX obo: <http://purl.obolibrary.org/obo/>
PREFIX up: <http://purl.uniprot.org/core/>


select distinct ?class ?classLabel ?property ?propertyLabel  where {

values ?class {
$$$
}

?instance a ?class.
optional {?class rdfs:label ?classLabel}
?instance ?property ?value
optional {?property rdfs:label ?propertyLabel}
filter isLiteral(?value) 
} '''

q3_label="Get all properties of some instances of a given concept"
q3='''
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX orth: <http://purl.org/net/orth#>
PREFIX genex: <http://purl.org/genex#>
PREFIX obo: <http://purl.obolibrary.org/obo/>
PREFIX up: <http://purl.uniprot.org/core/>

SELECT DISTINCT 
?class ?classLabel
?property ?propertyLabel 
WHERE {

{SELECT distinct ?instance {
?instance a $$$ .
 {$$$ a owl:Class} UNION {$$$ a rdfs:Class} }
  limit 100}

?instance a ?class .
optional {?class rdfs:label ?classLabel}
?instance ?property ?value
optional {?property rdfs:label ?propertyLabel}
filter isLiteral(?value)
} 
'''