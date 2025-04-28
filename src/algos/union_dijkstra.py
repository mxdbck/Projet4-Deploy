import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from problem import Problem
from typing import List

import scipy

from scipy.sparse.csgraph import shortest_path
import numpy as np

import random

def union_dijkstra(problem: Problem, dummy) -> List[bool]:
    wanted_journeys = problem.all_wanted_paths

    # Compute shortest paths: unpack the returned (dist_matrix, predecessors)
    dist_matrix, predecessors = shortest_path(
        problem.dist_mat,
        directed=True,
        return_predecessors=True,
        indices=[problem.airp_from_id(a).index for a in problem.wanted_sources],
    )

    solution = np.zeros(len(problem.all_connexions), dtype=bool)

    # Build a mapping from source airport id to the corresponding row in the predecessors matrix
    source_to_row = {a: i for i, a in enumerate(problem.wanted_sources)}

    # Iterate over each wanted journey (source, destination)
    for a, b in wanted_journeys:
        source_row = source_to_row[a]
        start_idx = problem.airp_from_id(a).index
        end_idx = problem.airp_from_id(b).index

        # Reconstruct the path from end_idx back to start_idx using the predecessor matrix
        path = [end_idx]
        while path[-1] != start_idx:
            path.append(predecessors[source_row, path[-1]])
        path.reverse()  # Reverse to get the path from start to end

        # Mark each connexion along the path in the solution vector
        for j in range(len(path) - 1):
            conn = (problem.airp_from_index(path[j]).id,
                    problem.airp_from_index(path[j+1]).id)
            solution[problem.connexion_index(conn)] = True

    return list(solution)


# Filter the chosen connexions to keep only the ones that are part of a t-spanner
# built by a greedy algorithm.
def t_spanner(problem: Problem, chosen_connexions: List[bool], t: float) -> List[bool]:
    # Build a list of (original_index, connexion) for each chosen connexion
    filtered_connexions = [
        (i, conn) for i, conn in enumerate(problem.all_connexions) if chosen_connexions[i]
    ]
    # Convert each connexion (id_u, id_v) to its (u, v) indices and keep the original index
    filtered_connexions_indices = [
        (orig_idx, (problem.airp_from_id(conn[0]).index, problem.airp_from_id(conn[1]).index))
        for orig_idx, conn in filtered_connexions
    ]

    # Sort the filtered connexions by the weight of the edge in the problem's distance matrix
    filtered_connexions_indices.sort(key=lambda x: problem.dist_mat[x[1][0], x[1][1]])

    # Create a working distance matrix based on the chosen connexions.
    n = len(problem.all_airports)
    dist_matrix = np.full((n, n), np.inf)
    for _, (u, v) in filtered_connexions_indices:
        dist_matrix[u, v] = problem.dist_mat[u, v]

    # For each connexion, try removing it and check if the remaining network still gives a t-spanner
    for orig_idx, (u, v) in filtered_connexions_indices:
        # Temporarily remove the edge
        saved_weight = dist_matrix[u, v]
        dist_matrix[u, v] = np.inf

        # Recompute all pairs shortest paths on the modified graph
        dijk = shortest_path(dist_matrix, directed=True, return_predecessors=False)

        # If the alternative path from u to v is within t times the direct edge weight,
        # then the edge is redundant and we mark it as not chosen.
        if dijk[u, v] <= t * saved_weight:
            chosen_connexions[orig_idx] = False
        else:
            # Otherwise, restore the removed edge.
            dist_matrix[u, v] = saved_weight

    return chosen_connexions

def t_spanner_full(problem: Problem, t: float) -> List[bool]:
    ones = list(np.ones(len(problem.all_connexions), dtype=bool))
    return t_spanner(problem, ones, t)

def union_spanner(problem: Problem, t: float) -> List[bool]:
    solution = union_dijkstra(problem, 100)
    return t_spanner(problem, list(solution), t)

import math
from cost import display_metrics, cost

def simulated_annealing(problem: Problem, initial_solution: List[bool], initial_temp: float, cooling_rate: float, max_iter: int):
    current_solution = initial_solution.copy()
    current_cost = cost(problem, current_solution)

    best_solution = current_solution.copy()
    best_cost = current_cost
    temperature = initial_temp

    print(f"SA Initial Temp={temperature:.2f}, Initial Cost={current_cost:.2f}")

    random.seed(69420)

    for i in range(max_iter):
        idx_mod = random.randint(0, len(current_solution) - 1)
        neighbor_solution = current_solution.copy()
        neighbor_solution[idx_mod] = not neighbor_solution[idx_mod]

        neighbor_cost = cost(problem, neighbor_solution)

        # Reject invalid neighbors
        if neighbor_cost == np.inf:
            acceptance_prob = 0
        else:
            delta_cost = neighbor_cost - current_cost
            if delta_cost < 0:
                acceptance_prob = 1.0 # Always accept improvements
            elif temperature > 1e-9:
                # Calculate probability for accepting worse solutions
                acceptance_prob = math.exp(-delta_cost / temperature)
            else:
                acceptance_prob = 0.0 # Temperature too low, only accept improvements

        if random.random() < acceptance_prob:
            current_solution = neighbor_solution
            current_cost = neighbor_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_solution = current_solution.copy()

        temperature *= cooling_rate

        if (i + 1) % 1000 == 0 or i == max_iter - 1: # Log every 1000 iters and at the end
             print(f"Iteration {i+1}: Temp={temperature:.4f}, Current Cost={current_cost:.2f}, Best Cost={best_cost:.2f}")
        elif temperature < 1e-6 and delta_cost >=0 and acceptance_prob < 1e-6:
             # Optional: Early exit if temperature is very low and no improvement is likely
             # print(f"Stopping early at iteration {i+1} due to low temperature and no recent improvement.")
             # break
             pass


    print(f"Finished SA. Total iterations: {max_iter}. Final best cost: {best_cost}")
    return list(best_solution)

def union_spanner_search(problem: 'Problem', t: float) -> List[bool]:
    return simulated_annealing(problem, union_spanner(problem, t), 1000.0, 0.99999, 100000)

def t_opt(c_values):
    optima = []
    for c_value in c_values:
        problem = import_csv()
        problem.c = c_value
        f = lambda t, problem: cost(problem, union_spanner(problem, t))
        optima.append(scipy.optimize.minimize(f, 1.1, args=(problem,), method='Nelder-Mead').x[0])
    return optima

from multiprocessing import Pool

def _calculate_cost_worker(t_value, problem_obj):
    current_cost = cost(problem_obj, union_spanner(problem_obj, t_value))
    return t_value, current_cost


def grid_search_t_optimizer(
    problem: 'Problem',
    min_t: float = 1.0,
    max_t: float = 2.0,
    num_steps: int = 100,
    num_cores: int = 8 # Added parameter for number of cores
) -> float:

    if min_t < 1.0:
        min_t = 1.0
    t_values = np.linspace(min_t, max_t, num_steps)

    print(f"Starting parallel grid search for t from {min_t:.4f} to {max_t:.4f} ({num_steps} steps) using {num_cores} cores...")
    tasks = [(t, problem) for t in t_values]

    results = []

    # Create a pool of worker processes
    with Pool(processes=num_cores) as pool:
        results = pool.starmap(_calculate_cost_worker, tasks)

    costs = np.array([res[1] for res in results])
    valid_indices = np.where(costs < sys.float_info.max)[0]

    min_cost_index = valid_indices[np.argmin(costs[valid_indices])]
    best_t = results[min_cost_index][0]
    min_cost = costs[min_cost_index]

    print(f"Grid search finished. Best t found: {best_t:.6f}, Minimum cost: {min_cost:.2f}")

    return best_t


if __name__ == "__main__":
    from utils.import_csv import import_csv
    c = 3500.0
    # problem = import_csv("./data/airports.csv", "./data/pre_existing_routes.csv", "./data/wanted_journeys.csv", c)
    problem = import_csv()
    solution = union_dijkstra(problem, 100)
    display_metrics(problem, list(solution), cost(problem, list(solution)), "Union Dijkstra")
    # solution2 = t_spanner(problem, list(solution), 1 + 0.4 * c /10000.0)
    solution2 = t_spanner(problem, list(solution), 1.14)
    # 1.5  => 60.25%
    # 1.4  => 60.70%
    # 1.35 => 60.25%
    # 1.3  => 60.45%
    display_metrics(problem, list(solution2), cost(problem, list(solution2)), "Union Dijkstra T-Spanner 1")
    solution3 = union_spanner(problem, 1.21)
    display_metrics(problem, list(solution3), cost(problem, list(solution3)), "Union Dijkstra T-Spanner 2")


    solution4 = simulated_annealing(problem, list(solution3), 1000.0, 0.99999, 100000)

    display_metrics(problem, list(solution4), cost(problem, list(solution4)), "Simulated Annealing")


    # # Plot optimum t vs cost
    # print("\nCalculating optimum t for different c values (this might take time)...")
    # # Reduce the number of points for faster execution during testing/debugging
    # # c_values = np.linspace(0, 100_000, 100)
    c_values = np.linspace(0, 20_000, 20) # Fewer points: start, end, num_points
    print(c_values)
    # # c_values = np.array([0, 1000, 3500, 10000, 25000, 50000]) # Specific values for testing
    # # c_values = np.array([3500.0]) # Single value test

    # optima = []
    # for c in c_values:
    #     problem = import_csv()
    #     problem.c = c
    #     optima.append(grid_search_t_optimizer(problem))
    # print("\nOptimal t values found (or attempted):")
    # print(list(optima)) # Convert numpy array (if returned) to list for printing

    # c_values_plot = c_values
    # # optima_plot = np.array([x[0] for x in optima])
    # optima_plot = np.array(optima)

    # # Determine Linear Fit
    # linear_fit = np.polyfit(c_values, optima_plot, 1)
    # print("\nLinear Fit Coefficients:")
    # print(linear_fit)

    # import matplotlib.pyplot as plt

    # plt.figure(figsize=(10, 6)) # Set figure size for better readability

    # plt.plot(c_values_plot, optima_plot)

    # plt.xlabel('Paramètre de coût fixe (c)')
    # plt.ylabel('Facteur d\'étirement optimal (t)')
    # plt.title('Valeur optimale de t en fonction du coût fixe c')


    # plt.grid(True, linestyle='--', alpha=0.7)

    # plt.tight_layout() # Adjust layout to prevent labels overlapping
    # plt.show()
