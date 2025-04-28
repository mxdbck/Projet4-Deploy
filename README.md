# Amélioration du trafic aérien international

A partir d'un réseau initial d'aéroports et de connexions, ce code optimise les connexions entre ces aéroports, sans créer de nouvelles infrastructures, tout en garantissant des trajets efficaces et viables pour les passagers et les compagnies aériennes.

### Prérequis

Assurez-vous que vous avez :
- Python 3.x installé
- Les dépendances suivantes :
    -geopy
    -networkx
    -pandas
    -streamlit
    -numpy
    -scipy
    -matplotlib

En utilisant le requirements.txt :
```bash
pip install -r requirements.txt
```

### Executer le projet

#### New Network

Pour obtenir notre nouveau réseau il faut appeler la fonction `new_network` dans le fichier `src/new_network.py`.
Celle-ci a comme arguments le chemin vers la liste des aéroports airports.csv , le chemin vers la liste des routes
déjà existantes pre_existing_routes.csv et le chemin vers la liste de trajets à satisfaire wanted_journeys.csv.
Le résultat `new_routes.csv` sera généré dans le dossier où l'utilisateur exécute le programme. Par exemple si
l'utilisateur se trouve dans le dossier racine du projet :
```bash
python src/new_network.py
```
et `new_routes.csv` sera généré dans le dossier racine du projet.

#### Interface

Pour lancer l'interface graphique il faut executer le fichier `interface.py` avec streamlit en s'assurant d'avoir
bien installé streamlit au préalable (`pip install streamlit` ou voir https://docs.streamlit.io/get-started/installation).
```bash
streamlit run src/interface.py
```
Une fenêtre s'ouvrira alors dans le navigateur par défaut de l'utilisateur (pour assurer la lisibilité il sera peut-être
nécessaire de modifier le thème en cliquant sur les trois points en haut à droite puis sur `Settings` puis
`Choose app theme, colors and fonts`).

#### Robustesse

Pour étudier la robustesse du graphe il faut executer le fichier `robustesse.py`.
IL génère un graphe du nombre d'arêtes gardées en fonction du pourcentage de robustesse désiré.
dans le dossier algorithme :
```bash
python src/algos/robustesse.py
```

#### Epidémie
Deux fichiers sont nécessaires pour analyser le réseau résilient face aux épidémies : epidemie.py, pour analyser la propagation d'un virus dans
le réseau, avec différents aéroports initialement infectés. Il y a également clustering_epidemie.py, qui permet de voir la valeur de la fonction coût ainsi que le temps moyen de propagation en fonction du nombre de clusters utilisés. Pour run ces deux fichiers et obtenir les graphes, il suffit simplement d'appuier sur le triangle en haut droite.

### Petit Bonus
Nous avons aussi implémenté un petit algorithme génétique qui recherche un réseau opimal. Pour l'exécuter, il faut simplementf aller dans :
```bash
python src/main_genetic.py
```
On peut personnaliser le réseau en modifiants les différents fichiers d'entrées spécifiés dans main_genetic.py .

## Contact

camille.leterme@student.uclouvain.be
jerome.dresse@student.uclouvain.be
julio.escobar@student.uclouvain.be
maxime.deboeck@student.uclouvain.be
w.kozlowska@student.uclouvain.be
