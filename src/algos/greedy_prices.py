import networkx as nx
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.import_csv import import_csv
from problem import Problem, Airport
from greedy_waiting_times import route_to_graph
from greed_prob import greed_prob


def find_cheapest_route(route, prob, start, end):
    start = prob.airp_from_id(start)
    end = prob.airp_from_id(end)
    graph = route_to_graph(route, prob)
    try:
        for u, v, data in graph.edges(data=True):
            price_key = (u.id, v.id)
            graph[u][v]['weight'] = prob.prices[price_key[0]][price_key[1]]
        path = nx.shortest_path(graph, source=start, target=end, weight='weight')
        routes_prises = [(path[i].id, path[i+1].id) for i in range(len(path) - 1)]
        routes_binary = [True if route in routes_prises else False for route in prob.all_connexions]
        path_ids = [node.id for node in path]
        total_cost = sum(graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
        return routes_binary, total_cost
    except nx.NetworkXNoPath:
        print(f"No path found from {start} to {end}.")
        return None, None

def test ():
    prob = import_csv("data/airports.csv", "data/pre_existing_routes.csv", "data/wanted_journeys_10.csv", 100.0)
    prices = pd.read_csv("data/prices.csv")
    H = greed_prob(prob, t=5)
    # Example usage
    file_path = "data/prices.csv"
    start_airport = prob.airp_from_id("AMS")
    end_airport = prob.airp_from_id("LIM")
    find_cheapest_route(H, prob, start_airport, end_airport)
