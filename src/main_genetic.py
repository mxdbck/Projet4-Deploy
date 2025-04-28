from algos.genetic import genetic_algorithm
from utils.import_csv import import_csv
import cProfile
import csv
import time
import matplotlib.pyplot as plt
from problem import create_problem_scc_alternative
from problem import create_problem


# TO MODIFY FOR PERSONALIZATION : 
airports_filename = "data/airports.csv"
pre_existing_routes_filename = "data/pre_existing_routes.csv"   
wanted_routes_filename = "data/wanted_journeys.csv"
prices_filename = "data/prices.csv"
waiting_times_filename = "data/waiting_times.csv"
c = 3500.0
# END TO MODIFY FOR PERSONALIZATION



problem = import_csv(
    airports_filename,
    pre_existing_routes_filename,
    wanted_routes_filename,
    prices_filename,
    waiting_times_filename,
    c
)


def csv_resultat(route):
    with open('resultat.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(route)



def plot_average_fitness(average_fitness):
    plt.figure()
    plt.plot(range(len(average_fitness)), average_fitness)
    plt.xlabel('Generation')
    plt.ylabel('Cost')
    plt.title('Cost over generations.png')
    plt.savefig('Cost_over_generations.png')
    #plt.show()

def plot_generations_time(generations_time):
    plt.figure()
    plt.plot(range(len(generations_time)), generations_time)
    plt.xlabel('Generation')
    plt.ylabel('Time (seconds)')
    plt.title('Time over generations')
    plt.savefig('Time_over_generations.png')
    #plt.show()



def execute_genetic (problem): 
    best_individual, average_fitness,generations_time = genetic_algorithm(problem)
    csv_resultat(best_individual)
    plot_average_fitness(average_fitness)
    plot_generations_time(generations_time)


if __name__ == "__main__":
    execute_genetic(problem)



    

    
    
