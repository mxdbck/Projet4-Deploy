import pandas as pd
from problem import Problem, Airport
import os

def import_csv(
    airports_filename = "./data/airports.csv",
    pre_existing_routes_filename = "./data/pre_existing_routes.csv",
    wanted_routes_filename = "./data/wanted_journeys.csv",
    prices_filename = "./data/prices.csv",
    waiting_times_filename = "./data/waiting_times.csv",
    c = 3500.0,
    ):
    """Importe les données des fichiers CSV et crée une instance de problème."""
    airports = pd.read_csv(airports_filename)
    airports = airports[["ID", "latitude", "longitude"]]
    routes = pd.read_csv(pre_existing_routes_filename)
    wanted_routes = pd.read_csv(wanted_routes_filename, header=None)
    wanted_routes = list(wanted_routes.itertuples(index=False, name=None))

    prices = pd.read_csv(prices_filename)

    # Turn columns 1 and 2 into keys for the dictionary
    # where the third column is the value
    prices = prices.set_index(["ID_start", "ID_end"])["price_tag"].unstack().to_dict()

    waiting_times = pd.read_csv(waiting_times_filename, dtype={'idle_time': float})
    # Turn the second column into the key for the dictionary
    # where the third column is the value
    waiting_times = waiting_times.set_index(["ID"])["idle_time"].to_dict()

    airport_list = []
    for i in range(len(airports)):
        airport_list.append(Airport(i, airports["ID"][i], (airports["latitude"][i], airports["longitude"][i]), waiting_times[airports["ID"][i]]))
        # print(waiting_times[airports["ID"][i]])
        # print(type(waiting_times[airports["ID"][i]]))
        # print(airport_list[i].simple_waiting_time)
        # print(type(airport_list[i].simple_waiting_time))

    problem_instance = Problem(airport_list, list(routes.itertuples(index=False, name=None)), list(wanted_routes), prices, c)
    # print(problem_instance.airp_from_id("MAD").simple_waiting_time)
    # print("dfghj")
    # print(problem_instance.airp_from_id("MAD").simple_waiting_time)
    # print(type(problem_instance.airp_from_id("MAD").simple_waiting_time))
    # print(problem_instance.connexion_id(4)[1])
    # print(type(problem_instance.airp_from_id(problem_instance.connexion_id(4)[1])))

    # print(type(problem_instance.airp_from_id(problem_instance.connexion_id(4)[1]).simple_waiting_time))

    return problem_instance


if __name__ == "__main__":
    import_csv("data/airports.csv", "data/pre_existing_routes.csv", "data/wanted_journeys.csv", "data/prices.csv", "data/waiting_times.csv", 100.0)
