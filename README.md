# Cy_Tech_Projet_PFE

Ce projet explore l'utilisation des GANs pour la génération de graphes complexes, avec une intégration d'un algorithme d'optimisation pour améliorer l'apprentissage du modèle.

Structure du projet:
1. generation_graphes_complexes
2. trouver_des_cliques_maximum

Ce README explique comment utiliser les scripts pour générer, entraîner et tester des graphes complexes via des GANs.

# Prérequis

Vous devez disposer de Python 3.x et des bibliothèques suivantes installées:
- tensorflow ou torch (selon votre implémentation GAN)
- networkx
- numpy
- matplotlib (pour la visualisation des graphes générés)

# Structure des dossiers

Le projet est divisé en deux dossiers principaux :
- generation_graphs_complexes : pour la génération des graphes.
- trouver_des_cliques_maximum : pour l'algorithme d'optimisation des cliques maximaux.

# Dossier "generation_graphes_complexes"

Ce dossier contient les scripts pour générer des graphes de différents types.

1.Pour créer un dataset de graphes, utilisez la commande suivante :
- python ./create_adjacency_matrices.py --graph_type <type_graphe>

Types de graphes :
- 1 -> Graphe planaire
- 2 -> Graphe cactus
- 3 -> Graphe cycle
- 4 -> Graphe arbre
- 5 -> Graphe arbre binaire

2.Entraînement du modèle GAN

Une fois le dataset de graphes généré, vous pouvez entraîner le modèle GAN avec la commande suivante :
- python ./training.py --type_graphs <type_graphe>

# Dossier "trouver_des_cliques_maximum"
Ce dossier contient l'algorithme d'optimisation utilisé pour détecter les cliques maximaux dans les graphes générés.
