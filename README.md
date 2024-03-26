# Titre du Projet : Visualisation de systèmes complexes 

## Description

Ce projet a pour but d'implémenter un visualisateur de système complexes tel que le jeu de la vie ou Lenia/Flow Lenia pour le visualiser de différentes manières en 2 et principalement 3 dimensions.

## Fonctionnalités

- Visualiser la simulation de plusieurs systèmes complexes sur la même interface,
- Configurer l'initialisation et les règles du système complexe avant et pendant la simulation
- Configurer les paramètres de visualisation (couleur, position, rotation, taille...)
- Intéragir avec le système complexe et voir visuellement le résultat des actions effectuées à la souris. 

## Dépendances

- Python 3.10
- `Django`, `Channels` ,`Daphne`, `Jax`, `Jaxlib`, `jax-md`, 
- WebGL 2.0

## Installation

Étapes pour installer les dépendances nécessaires et pour exécuter le projet :

```bash
git clone https://github.com/BastienMrl/Complex-systems-visualization.git
cd Complex-systems-visualization
pip install -r requirements.txt
npm install -g typescript #optionnel (pour modifier les scripts typescript puis les recompiler en javascript)
```

## Utilisation

Lancer le serveur en local :

```bash
cd complexSystemViewer
tsc #Compilation optionelle du typescript en javascript (a faire seulement si les scripts typescript ont été modifiés)
python manage.py runserver
```

## Démonstrations 

Démonstration générale du projet : 

https://youtu.be/2VxnvdJWnyQ?si=tYZlP-0knwq7luYX

Playlist avec d'autres vidéos sur différents systèmes complexes :

https://youtube.com/playlist?list=PLicm65r9hfEL8m3mh3B4AFHPbdZ8ZIOW9&si=NEcGsk8mqDfWtVh0

![Exemple de capture d'écran](lien_vers_capture.png)


## Licence

A définir..

## Auteurs

- Victor Barrey
- Léo Dupouey
- Bastien Morel
- Moze Jonathan
