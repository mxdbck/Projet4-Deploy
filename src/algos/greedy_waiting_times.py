import networkx as nx
from geopy.distance import geodesic
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from problem import Airport

def route_to_graph(route, prob):
    G = nx.DiGraph()
    pre_existing_routes = prob.all_connexions
    for i, (start_id, end_id) in enumerate(pre_existing_routes):
        if route[i]:
            u = prob.airp_from_id(start_id)
            v = prob.airp_from_id(end_id)
            distance = geodesic((u.location), (v.location)).km
            G.add_edge(u, v, weight=distance)
    return G



def waiting_times(route, prob, start_airport, end_airport) :
    start_airport = prob.airp_from_id(start_airport)
    end_airport = prob.airp_from_id(end_airport)
    H = route_to_graph(route, prob)
    wait = pd.read_csv("data/waiting_times.csv")
    wait = wait.set_index("ID")
    for u, v, d in H.edges(data=True) :
        d['weight'] = (d['weight'] / 900) * 60
    for node in H.nodes():
        waiting_time = wait.loc[node.id, "idle_time"] / 2
        for neighbor in H.neighbors(node):
            if H.has_edge(node, neighbor):
                H[node][neighbor]['weight'] += waiting_time
            if H.has_edge(neighbor, node):
                H[neighbor][node]['weight'] += waiting_time

    # Create START and END airports
    START = Airport(100, "START", (0.0, 0.0),0)
    END = Airport(101, "END", (0.0, 0.0),0)
    H.add_node(START)
    H.add_edge(START, start_airport, weight=-wait.loc[start_airport.id, "idle_time"] / 2)

    H.add_node(END)
    H.add_edge(end_airport, END, weight=-wait.loc[end_airport.id, "idle_time"] / 2)
    try : 
        path = nx.shortest_path(H, source=START, target=END, weight='weight')
    except nx.NetworkXNoPath:
        return None, None
    routes_prises = [(path[i].id, path[i+1].id) for i in range(len(path) - 1)]
    routes_binary = [True if route in routes_prises else False for route in prob.all_connexions]
    total_travel_time = sum(H[path[i]][path[i+1]]['weight'] for i in range(len(path) - 1))
    hours, minutes = divmod(total_travel_time, 60)
    total_travel_time_str = f"{int(hours)} : {int(minutes)}"
    path = path[1:-1]
    path_ids = [node.id for node in path]
    routes = [(path[i].id, path[i+1].id) for i in range(len(path) - 1)]
    return routes_binary, total_travel_time


"""prob = import_csv("data/airports.csv", "data/pre_existing_routes.csv", "data/wanted_journeys.csv", 100.0)


waiting_times("greedy",prob, prob.airp_from_id('CKG'), prob.airp_from_id('SEA'))"""