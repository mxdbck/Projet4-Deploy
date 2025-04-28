from typing import List, Tuple, Dict
from geopy.distance import geodesic
import numpy as np

class Airport:
    def __init__(self, index: int, id: str, location: Tuple[float, float], waiting: float):
        self.index = index
        self.id = id
        self.location = location
        self.simple_waiting_time = waiting

    def distance_from(self, other: 'Airport') -> float:
        return geodesic(self.location, other.location).km

class Problem:
    def __init__(
            self,
            airports: List[Airport],
            connexions: List[Tuple[str, str]],
            paths: List[Tuple[str, str]],
            prices: Dict[Tuple[str, str], float],
            c: float
        ):
        self.all_airports: List[Airport] = airports
        self.n_airports: int = len(airports)
        # --- Modified: Create a dictionary for fast airport lookup ---
        self.airport_dict = {airport.id: airport for airport in airports}

        self.all_connexions: List[Tuple[str, str]] = connexions
        self.prices: Dict[Tuple[str, str], float] = prices
        self.all_wanted_paths: List[Tuple[str, str]] = paths
        self.dist_mat: np.ndarray = np.ones((len(airports), len(airports))) * np.inf
        self.c = c

        self.plane_speed = 900

        self.wanted_sources = []
        self.wanted_destinations = []
        for path in paths:
            if path[0] not in self.wanted_sources:
                self.wanted_sources.append(path[0])
            if path[1] not in self.wanted_destinations:
                self.wanted_destinations.append(path[1])

        # Numpy versions for cost calculation
        self.connexions = np.array(self.all_connexions)
        self.indices_a = np.array([self.airport_dict[a].index for a in self.connexions[:, 0]])
        self.indices_b = np.array([self.airport_dict[a].index for a in self.connexions[:, 1]])

        # --- Modified: Use cached dictionary for lookups ---
        for a, b in self.all_connexions:
            self.dist_mat[self.airport_dict[a].index, self.airport_dict[b].index] = self.airport_dict[a].distance_from(self.airport_dict[b])

        for i in range(len(self.all_airports)):
            self.dist_mat[i, i] = 0

        self.naive_cost = 0.0

    # --- Modified: airp_from_id now uses the dictionary ---
    def airp_from_id(self, id: str) -> Airport:
        if id in self.airport_dict:
            return self.airport_dict[id]
        else:
            raise ValueError(f"Airport with id {id} not found")

    def airp_from_index(self, index: int) -> Airport:
        return self.all_airports[index]

    def connexion_index(self, connexion: Tuple[str, str]) -> int:
        return self.all_connexions.index(connexion)

    def connexion_id(self, index: int) -> Tuple[str, str]:
        return self.all_connexions[index]

    def path_index(self, path: Tuple[str, str]) -> int:
        return self.all_wanted_paths.index(path)

    def path_id(self, index: int) -> Tuple[str, str]:
        return self.all_wanted_paths[index]

def get_prices(problem: Problem, path: List[bool], ) -> float:
    result = 0.0
    for i in range(len(path)):
        if path[i]:
            entrant, sortant = problem.connexion_id(i)
            # print(problem.prices[entrant][sortant])
            result += float(problem.prices[entrant][sortant])
    return result

def get_distance(problem: Problem, path: List[bool], start, end) -> float:
    result = 0
    for i in range(len(path)):
        # if path[i] and i != start and i != end:
        if path[i]:
            result += problem.airp_from_id(problem.connexion_id(i)[0]).distance_from(problem.airp_from_id(problem.connexion_id(i)[1]))
    return result


def get_waiting_times(problem: Problem, path: List[bool], start, end) -> float:
    result = 0.0
    print(f"Start and end {start}: {end}")
    for i in range(len(path)):
        if path[i] and i != start and i != end:
            # Debugging: print connection details
            print(f"Processing connection index {i}: {problem.connexion_id(i)}")


            airport = problem.airp_from_id(problem.connexion_id(i)[0])
            if (airport.id != start and airport.id != end ):
                print(f"Airport {airport.id}: Waiting time = {airport.simple_waiting_time}")
                result += airport.simple_waiting_time  # Ensure it's a float

            # Calculate distance-based waiting time
            next_airport = problem.airp_from_id(problem.connexion_id(i)[1])
            distance = airport.distance_from(next_airport)
            waiting_time = 60 * distance / problem.plane_speed

            print(f"Distance: {distance}, Extra waiting time: {waiting_time}")

            result += waiting_time  # Add distance-based waiting time

    return result




def create_problem(number_of_airports: int, number_of_connexions: int, number_of_paths: int, c: float, seed: int = 42) -> Problem:
    import random
    random.seed(seed)

    # Create airports with random coordinates.
    airports = [
        Airport(i, f"ID{i}", (random.uniform(-900, 900) / 10, random.uniform(-1800, 1800) / 10), random.uniform(50, 300))
        for i in range(number_of_airports)
    ]

    # Build a spanning tree to guarantee connectivity.
    airport_ids = [airport.id for airport in airports]
    random.shuffle(airport_ids)
    connexions = []
    # For each undirected edge, add both directions.
    for i in range(1, len(airport_ids)):
        a, b = airport_ids[i - 1], airport_ids[i]
        connexions.append((a, b))
        connexions.append((b, a))

    # If more connections are needed, add extra random bidirectional edges.
    while len(connexions) < number_of_connexions:
        a, b = random.choice(airport_ids), random.choice(airport_ids)
        # Ensure we don't add self loops or duplicate undirected edges.
        if a != b and ((a, b) not in connexions and (b, a) not in connexions):
            connexions.append((a, b))
            connexions.append((b, a))

    # Generate wanted paths (they can remain as one-directional pairs if that's acceptable).
    wanted_paths = [(random.choice(airport_ids), random.choice(airport_ids)) for _ in range(number_of_paths)]

    # Generate Prices
    prices = {conn: random.uniform(100, 2000) for conn in connexions}

    return Problem(airports, connexions, wanted_paths, prices, c)

def create_problem_scc_alternative(number_of_airports: int, number_of_connexions: int, number_of_paths: int, c: float, seed: int = 42) -> Problem:
    import random
    random.seed(seed)

    # Create airports with random coordinates.
    airports = [
        Airport(i, f"ID{i}", (random.uniform(-900, 900) / 10, random.uniform(-1800, 1800) / 10), random.uniform(50, 300))
        for i in range(number_of_airports)
    ]
    airport_ids = [airport.id for airport in airports]

    # Initially, treat every airport as its own SCC.
    components = [[aid] for aid in airport_ids]
    merge_edges = []  # This will store the "merging" edges needed to ensure strong connectivity.

    # Merge components until a single SCC remains.
    while len(components) > 1:
        # Randomly pick two distinct components.
        i = random.randrange(len(components))
        j = random.randrange(len(components))
        while j == i:
            j = random.randrange(len(components))
        comp1 = components[i]
        comp2 = components[j]

        # For the merge, pick one random node from each component.
        # First, add an edge from a node in comp1 to a node in comp2.
        node_from_comp1 = random.choice(comp1)
        node_to_comp2 = random.choice(comp2)
        merge_edges.append((node_from_comp1, node_to_comp2))

        # Later, add an edge in the reverse direction.
        node_from_comp2 = random.choice(comp2)
        node_to_comp1 = random.choice(comp1)
        merge_edges.append((node_from_comp2, node_to_comp1))

        # Merge the two components into one.
        new_component = comp1 + comp2
        # Remove the two components (remove the one with higher index first).
        if i > j:
            components.pop(i)
            components.pop(j)
        else:
            components.pop(j)
            components.pop(i)
        components.append(new_component)

    # Shuffle the merge edges so that the reverse edge doesn't immediately follow its counterpart.
    random.shuffle(merge_edges)
    connexions = merge_edges.copy()  # Start with the merge edges.

    # If extra connections are desired, add additional single directed edges.
    while len(connexions) < number_of_connexions:
        a, b = random.choice(airport_ids), random.choice(airport_ids)
        # Ensure we don't add self-loops and avoid duplicate edges.
        if a != b and ((a, b) not in connexions):
            connexions.append((a, b))

    # If we accidentally overshoot the number, truncate (ideally, number_of_connexions is at least 2*(n-1)).
    connexions = connexions[:number_of_connexions]

    # Generate wanted paths as random directed pairs.
    wanted_paths = [(random.choice(airport_ids), random.choice(airport_ids)) for _ in range(number_of_paths)]

    prices = {(a, b): random.uniform(10, 2000) for a, b in connexions}

    return Problem(airports, connexions, wanted_paths, prices, c)


if __name__ == "__main__":
    problem = create_problem(100, 2000, 50, 10)

    import cost

    cost.display_metrics(problem, [True] * len(problem.all_connexions), 0, "Test")
