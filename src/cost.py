import numpy as np
from scipy.sparse.csgraph import shortest_path
from typing import List, Tuple
from problem import Problem

# from cost_wrapper import cost_optimized as cost_optimized2

from scipy.sparse import coo_matrix, csr_matrix

def cost(problem: Problem, binary_selection: List[bool]) -> float:
    """Calcul du coût total."""
    # Création de la matrice d'adjacence pondérée
    # connexions = np.array(problem.all_connexions)
    # indices_a = np.array([problem.airport_dict[a].index for a in connexions[:, 0]])
    # indices_b = np.array([problem.airport_dict[a].index for a in connexions[:, 1]])

    # Ajout des connexions sélectionnées
    selected_indices = np.where(binary_selection)[0]

    # Mode dense:
    # adj = np.zeros((problem.n_airports, problem.n_airports), dtype=np.float32)
    # adj[problem.indices_a[selected_indices], problem.indices_b[selected_indices]] = problem.dist_mat[problem.indices_a[selected_indices], problem.indices_b[selected_indices]]

    # Mode sparse:
    rows = problem.indices_a[selected_indices]
    cols = problem.indices_b[selected_indices]
    data = problem.dist_mat[rows, cols]
    adj = csr_matrix((data, (rows, cols)), shape=(problem.n_airports, problem.n_airports))


    # Calcul des plus courts chemins (Dijkstra)
    source_indices = [problem.airport_dict[a].index for a in problem.wanted_sources]
    distances = shortest_path(
        adj,
        directed=True,
        return_predecessors=False,
        indices=source_indices,
    )

    # Création d'un mapping source -> ligne dans la matrice des distances
    source_to_row = {a: i for i, a in enumerate(problem.wanted_sources)}

    # Coût du au nombre de connexions choisies
    cost_sum = np.count_nonzero(binary_selection) * problem.c

    # Coût du à la distance des chemins voulus
    start_idx, end_idx = [], []
    for (src_airport, dst_airport) in problem.all_wanted_paths:
        start_idx.append(source_to_row[src_airport])
        end_idx.append(problem.airport_dict[dst_airport].index)
    cost_sum += np.sum(distances[start_idx, end_idx])


    return cost_sum


def cost_genetic(problem: Problem, binary_selection: List[bool], genetic = True) -> float:
    """Calcul du coût total."""
    # Création de la matrice d'adjacence pondérée
    # connexions = np.array(problem.all_connexions)
    # indices_a = np.array([problem.airport_dict[a].index for a in connexions[:, 0]])
    # indices_b = np.array([problem.airport_dict[a].index for a in connexions[:, 1]])

    # Ajout des connexions sélectionnées
    selected_indices = np.where(binary_selection)[0]

    # Mode dense:
    # adj = np.zeros((problem.n_airports, problem.n_airports), dtype=np.float32)
    # adj[problem.indices_a[selected_indices], problem.indices_b[selected_indices]] = problem.dist_mat[problem.indices_a[selected_indices], problem.indices_b[selected_indices]]

    # Mode sparse:
    rows = problem.indices_a[selected_indices]
    cols = problem.indices_b[selected_indices]
    data = problem.dist_mat[rows, cols]
    adj = csr_matrix((data, (rows, cols)), shape=(problem.n_airports, problem.n_airports))


    # Calcul des plus courts chemins (Dijkstra)
    source_indices = [problem.airport_dict[a].index for a in problem.wanted_sources]
    distances = shortest_path(
        adj,
        directed=True,
        return_predecessors=False,
        indices=source_indices,
    )



    # Création d'un mapping source -> ligne dans la matrice des distances
    source_to_row = {a: i for i, a in enumerate(problem.wanted_sources)}

    # Coût du au nombre de connexions choisies
    cost_sum = np.count_nonzero(binary_selection) * problem.c

    # Coût du à la distance des chemins voulus
    start_idx, end_idx = [], []
    for (src_airport, dst_airport) in problem.all_wanted_paths:
        start_idx.append(source_to_row[src_airport])
        end_idx.append(problem.airport_dict[dst_airport].index)
    cost_sum += np.sum(distances[start_idx, end_idx])

    if genetic:
        if np.isinf(cost_sum):
            cost_sum = problem.naive_cost * 2.0
    return cost_sum


def is_connex(problem: Problem, routes: List[Tuple[str, str]]) -> bool:
    """Vérifie si le graphe est connexe."""
    n = len(problem.all_airports)
    adj = np.zeros((n, n), dtype=np.float32)

    # Création rapide de la matrice d'adjacence
    indices = np.array([(problem.airp_from_id(a).index, problem.airp_from_id(b).index) for a, b in routes])
    adj[indices[:, 0], indices[:, 1]] = 1

    # Vérification de la connexité
    distances = shortest_path(adj, directed=True, return_predecessors=False)

    return not np.any(distances == np.inf)


def display_metrics(problem: Problem, binary_selection: List[bool], final_cost: float, algo_name: str):
    """Affichage des métriques du problème."""
    n_airports, n_connexions, n_paths = map(len, [problem.all_airports, problem.all_connexions, problem.all_wanted_paths])
    n_selected = np.count_nonzero(binary_selection)

    naive_cost = cost(problem, [True] * n_connexions)
    reduction = 100 * (naive_cost - final_cost) / naive_cost if naive_cost else 0

    num_connexions_cost = np.count_nonzero(binary_selection) * problem.c
    average_distance_cost = final_cost - num_connexions_cost

    naive_connexions_cost = n_connexions * problem.c
    naive_distance_cost = naive_cost - naive_connexions_cost

    # Affichage
    print("-" * 60)
    print(f"{'Problem Metrics'.center(60)}")
    print("-" * 60)
    print(f"{'Airports':<25}: {n_airports:,}")
    print(f"{'Possible Connections':<25}: {n_connexions:,}")
    print(f"{'Wanted Paths':<25}: {n_paths:,}")
    print(f"{'c (cost per connection)':<25}: {problem.c}")
    print("-" * 60)
    print(f"{'Algorithm':<25}: {algo_name}")
    print(f"{'Naive Cost':<25}: {naive_cost:,.2f} (using all {n_connexions:,} connections)")
    print(f"{'Naive Con. Cost':<25}: {naive_connexions_cost:,.2f}")
    print(f"{'Naive Dist. Cost':<25}: {naive_distance_cost:,.2f}")
    print(f"{'Optimal Cost':<25}: {final_cost:,.2f} (using {n_selected:,} connections)")
    print(f"{'N Connections Cost':<25}: {num_connexions_cost:,.2f}")
    print(f"{'Average Distance Cost':<25}: {average_distance_cost:,.2f}")
    print(f"{'Reduction':<25}: {reduction:,.2f}%")
    print("-" * 60)
