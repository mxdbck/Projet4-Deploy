import networkx as nx
from geopy.distance import geodesic
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.import_csv import import_csv

"""
Algorithme glouton, utilisant la classe Problem
"""

prob = import_csv()

def greedy_spanner(prob, t=2):
    """ Algorithme glouton pour la construction d'un t-spanner. """
    # Construire le graphe initial avec poids
    G = nx.DiGraph()
    G.add_nodes_from(prob.all_airports)
    for u in prob.all_airports:
        for v in prob.all_airports:
            if u != v:
                if (u.id, v.id) in prob.all_connexions:
                    weight = geodesic(u.location, v.location).km
                    G.add_edge(u, v, weight=weight)

    # Trier les arêtes par poids croissant
    sorted_edges = sorted(G.edges(data=True), key=lambda x: x[2]["weight"])

    # Initialiser le sous-graphe H vide
    H = nx.DiGraph()
    H.add_nodes_from(prob.all_airports)

    for u, v, data in sorted_edges:
        weight = data["weight"]
        if not H.has_node(u) or not H.has_node(v):
            continue

        # Vérifier la condition du t-spanner
        if nx.has_path(H, u, v):
            shortest_path_length = nx.shortest_path_length(H, u, v, weight="weight")
            if shortest_path_length <= t * weight:
                continue

        # Ajouter l'arête si la contrainte est respectée
        H.add_edge(u, v, weight=weight)

    route_gardees = []

    for u in prob.all_wanted_paths:
        for v in prob.all_wanted_paths:
            if u != v:
                shor_path = nx.shortest_path(H, prob.airp_from_id(u[0]), prob.airp_from_id(v[1]), weight="weight")
                for i in range(len(shor_path) - 1):
                    edge = (shor_path[i].id, shor_path[i + 1].id)
                    if edge not in route_gardees:
                        route_gardees.append(edge)
    arete_tot = list(H.edges)
    arete_tot = [(u.id, v.id) for u, v in arete_tot]
    for route in arete_tot :
        if route not in route_gardees:
            H.remove_edge(prob.airp_from_id(route[0]), prob.airp_from_id(route[1]))
    return H


def greed_prob(prob, t=5):
    H = greedy_spanner(prob, t)
    routes = [(u.id, v.id) for u, v in H.edges()]
    routes_binary = [True if route in routes else False for route in prob.all_connexions]
    return routes_binary
