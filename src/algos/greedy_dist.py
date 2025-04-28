import networkx as nx
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from greedy_waiting_times import route_to_graph

def find_shortest_distance(route, prob, start, end):
    start = prob.airp_from_id(start)
    print(type(start))
    end = prob.airp_from_id(end)
    # Find the cheapest path
    try:
        graph = route_to_graph(route, prob) 
        path = nx.shortest_path(graph, source=start, target=end, weight='weight')
        routes_prises = [(path[i].id, path[i+1].id) for i in range(len(path) - 1)]
        routes_binary = [True if route in routes_prises else False for route in prob.all_connexions]
        path_ids = [node.id for node in path]
        total_distance = sum(graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
        return routes_binary, total_distance
    except nx.NetworkXNoPath:
        print(f"No path found from {start} to {end}.")
        return None, None
"""
# Example usage
# problem is an object with attributes distance_matrix and aeroports
prob = import_csv("data/airports.csv", "data/pre_existing_routes.csv", "data/wanted_journeys.csv", 100.0)
start_airport = prob.airp_from_id("HRG")
end_airport = prob.airp_from_id("PHX")
H = greedy_spanner(prob, t=5)
find_shortest_distance(H, start_airport, end_airport)"""
