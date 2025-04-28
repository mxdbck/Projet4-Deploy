import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from greedy_dist import find_shortest_distance
from greedy_prices import find_cheapest_route
from greedy_waiting_times import waiting_times
from utils.import_csv import import_csv
from union_dijkstra import union_dijkstra, union_spanner, union_spanner_search, t_spanner_full
import csv
from problem import get_prices
from problem import get_waiting_times
from problem import get_distance

import time
import numpy as np
import pickle

prob = import_csv()

def find_proprietes_du_trajet(prob, start, end, algo="union-spanner-search"):

    routes = None
    cache_filename = "./data/union_spanner_search_cache.pkl"
    use_cache = (algo == "union-spanner-search")

    t0 = time.time()
    if algo == "t-spanner":
        routes = t_spanner_full(prob, t=1.1 + 0.25*3500.0 / 10_000.0)
    elif algo == "union":
        routes = union_dijkstra(prob, 0)
    elif algo == "union-spanner" :
        routes = union_spanner(prob, 1.1 + 0.25*3500.0 / 10_000.0 )
    elif algo == "union-spanner-search" :
        # --- Caching Logic ---
        if use_cache and os.path.exists(cache_filename):
            print(f"Attempting to load cached results from: {cache_filename}")
            try:
                with open(cache_filename, 'rb') as f: # Open in binary read mode
                    routes = pickle.load(f)
                print("Successfully loaded routes from cache.")
            except Exception as e:
                print(f"Error loading cache file {cache_filename}: {e}. Recomputing...")
                routes = None # Ensure computation happens if loading fails

        # Compute if cache wasn't used, didn't exist, or failed to load
        if routes is None:
            print("Computing routes using union_spanner_search...")
            routes = union_spanner_search(prob, 1.1 + 0.25*3500.0 / 10_000.0 )
            # Save the result to cache if caching is enabled for this algo
            if use_cache:
                print(f"Saving computed routes to cache: {cache_filename}")
                try:
                    with open(cache_filename, 'wb') as f: # Open in binary write mode
                        pickle.dump(routes, f)
                except Exception as e:
                    print(f"Error saving cache file {cache_filename}: {e}")
        # --- End Caching Logic ---
    elif algo == "all":
        routes = list(np.ones(len(prob.all_connexions)))
    t1 = time.time()

    print("Temps d'exécution de l'algorithme : ", t1-t0)


    # 1 = shortest distance
    # 2 = cheapest route
    # 3 = waiting times

    path = [0,0,0]
    path_binaire = [0,0,0]
    nombre_escales = [0,0,0]
    distances = [0,0,0]
    costs = [0,0,0]
    travel_times = [0,0,0]
    escales = [0,0,0]

    t0 = time.time()

    path_binaire[0], distances[0] = find_shortest_distance(routes, prob, start, end)
    if path_binaire[0] == None:
        return None, None, None, None, None, None, None

    costs[0] = get_prices(prob, path_binaire[0])
    travel_times[0] = get_waiting_times(prob, path_binaire[0], start, end)

    path_binaire[1], costs[1] = find_cheapest_route(routes, prob, start, end)
    print("Cost", costs[1])
    distances[1] = get_distance(prob, path_binaire[1], start, end)
    travel_times[1] = get_waiting_times(prob, path_binaire[1], start, end)
    costs[1] = get_prices(prob, path_binaire[1])
    print("cost", costs[1])

    path_binaire[2], travel_times[2] = waiting_times(routes, prob, start, end)
    distances[2] = get_distance(prob, path_binaire[2], start, end)
    costs[2] = get_prices(prob, path_binaire[2])
    travel_times[2] = get_waiting_times(prob, path_binaire[2], start, end)
    if travel_times[2] > travel_times[1] or travel_times[2] > travel_times[0]:
        to_take = min(travel_times[1], travel_times[0])
        idx = travel_times.index(to_take)
        path_binaire[2] = path_binaire[idx]
        travel_times[2] = travel_times[idx]
        distances[2] = distances[idx]
        costs[2] = costs[idx]


    t1 = time.time()
    print("Temps d'exécution des calculs : ", t1-t0)

    t0 = time.time()

    for i in range(3):
        path[i],escales[i],nombre_escales[i] = get_airports_codes_in_order(prob, path_binaire[i], start, end)
        travel_times[i] =float( travel_times[i])
        costs[i] =float( costs[i])


    return path, escales, distances, costs, travel_times, nombre_escales, path_binaire

def get_airports_codes_in_order(prob, path, start, end):
    # List to hold the airport codes in the correct order
    if (type(start) != str):
        print("WTF")
        start = start.id
        end = end.id
    # print("start et end")
    # print(type(start))
    # print(type(end))
    airports_codes_in_order = []
    path_code = [start]

    # Add the start node's airport code
    #airports_codes_in_order.append(start)

    # We will start from the start node and look for connections
    current_node = start

    # Iterate over all connections and follow the correct path
    while (current_node != end ):
        for i, is_selected in enumerate(path):
            if is_selected :
                start_id, end_id = prob.connexion_id(i)

                # Check if the current node matches the start of the connection
                if current_node == start_id:
                    # Add the end airport code
                    if (end_id != end) :
                        airports_codes_in_order.append(end_id)
                    # Update the current node
                    current_node = end_id
                    break


    nombre_escales = len(airports_codes_in_order)
    if nombre_escales==0 :
        path_code += [end]
        return path,' - ',nombre_escales

    # Join the list of airport codes into a string separated by ' - '
    path_code += airports_codes_in_order + [end]
    return path_code,' - '.join(airports_codes_in_order),nombre_escales


def csv_resultat(route):
    with open('resultat.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(route)

#result = greed_prob(prob, t=5)
#csv_resultat(result)
#print(result)
