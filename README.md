# Search-Engine (Recherche d'information):

## Contexte :

* Ce répo présente une implémntation d'un moteur de recherche  à partir de zéro en utilisant le paradigme POO pour la modularité, la réutilisabilité et la clarté.
* Il est basé sur de nombreuses fonctions génériques utilisant différents paramètres inspirés de la signature des fonctions scikit-learn.

## Objectif :

* Comprendre les pricipes et les concepts de la recherche d'information, et les implémenter dans le cadre d'un cas d'usage (moteur de recherche).

## Doonées : 

Disponilbel sous le répertoire **./data**. Les dataset sont organisés sous forme de collection de documents et de requetes .

* CACM.
* CISI.

## Fonctionalitées implémentées :

- Documents Parser ( Simple | Regex ) 
- Queries Parser
- Indexation ( normal index , inverted index )
- Weighters

- Models : 
  
  1. Vectorial
  2. Language model
  3. Okapi-BM25 

- Métrics d'évaluation :
  1. Precision
  2. Recall
  3. F-measure
  4. Average Precision ( AvgP )
  5. Mean Average Precision ( MAP )
  6. Mean Reciprocal Rank ( MRR )
  7. Normalized Discounted cumulative gain ( NDCG )

- Models Evaluation Platform and Reporting.