import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from problem import Problem
from typing import List

from scipy.sparse.csgraph import shortest_path
import numpy as np


def union_dijkstra(problem: Problem, dummy) -> tuple[np.ndarray, bool]:
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

    path_was_disconnected = False

    # Iterate over each wanted journey (source, destination)
    for a, b in wanted_journeys:
        source_row = source_to_row[a]
        start_idx = problem.airp_from_id(a).index
        end_idx = problem.airp_from_id(b).index

        # Reconstruct the path from end_idx back to start_idx using the predecessor matrix
        path = [end_idx]
        flag = False
        while path[-1] != start_idx:
            # print(source_row, path[-1])
            if path[-1] == -9999:
                flag = True
                path_was_disconnected = True
                break
            path.append(predecessors[source_row, path[-1]])
        if flag:
            continue
        path.reverse()  # Reverse to get the path from start to end

        # Mark each connexion along the path in the solution vector
        for j in range(len(path) - 1):
            conn = (problem.airp_from_index(path[j]).id,
                    problem.airp_from_index(path[j+1]).id)
            solution[problem.connexion_index(conn)] = True

    return solution, path_was_disconnected # vaut False si tous les chemins sont trouvés, True sinon


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

def union_spanner(problem: Problem, t: float) -> tuple[List[bool], bool]:
    solution, booleen = union_dijkstra(problem, 100)
    return t_spanner(problem, list(solution), t), booleen

if __name__ == "__main__":
    from utils.import_csv import import_csv
    from cost import display_metrics, cost
    c = 50000.0
    problem = import_csv("data/airports.csv", "data/pre_existing_routes.csv", "data/wanted_journeys.csv", c)
    solution = union_dijkstra(problem, 100)
    display_metrics(problem, list(solution), cost(problem, list(solution)), "Union Dijkstra")
    solution2 = t_spanner(problem, list(solution), 1 + 0.4 * c /10000.0)
    # 1.5  => 60.25%
    # 1.4  => 60.70%
    # 1.35 => 60.25%
    # 1.3  => 60.45%
    display_metrics(problem, list(solution2), cost(problem, list(solution2)), "Union Dijkstra T-Spanner")
