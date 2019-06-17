# -*- coding: utf-8 -*-
"""
Created on Tue May 21 20:10:37 2019

@author: Jakub Grzeszczak
"""

import numpy  as np
import matplotlib.pyplot as plt
import copy
import random
import os
from mpl_toolkits.mplot3d.axes3d import get_test_data
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


class SPEA2:
    class solution:
        def __init__(self, var_bounds, objective_functions):
            self.variables = [random.uniform(bounds[0], bounds[1]) for bounds in var_bounds]
            self.objectives = [function(self.variables) for function in objective_functions]
    
    def dominated_by(self, sol_1, sol_2):
        for i in range(self.objective_functions_dimensions):
            if sol_2.objectives[i] > sol_1.objectives[i]:
                return False
        return True
    
    def calculate_raw_fitness(self, population):
        strength = np.zeros(self.length)
        raw_fitness = np.zeros(self.length)
        dominations = {}
        for i in range(self.length):
            dominations[i] = []
            for j in range(self.length):
                if i != j:
                    if not self.dominated_by(population[i], population[j]):
                        strength[i] = strength[i] + 1
                    else:
                        dominations[i].append(j)
        for i in range(self.length):
            for j in dominations[i]:
                raw_fitness[i] = raw_fitness[i] + strength[j]
        return raw_fitness
    
    def calculate_fitness_and_distances(self, population, raw_fitness):
        k_n = int(self.length**0.5) - 1
        fitness = np.zeros(self.length)
        all_distances = []
        for i in range(self.length):
            distances = []
            for j in range(self.length):
                if i != j:
                    sol_1 = population[i]
                    sol_2 = population[j]
                    distance = sum([(sol_1.objectives[i] - sol_2.objectives[i])**2 for i in range(self.objective_functions_dimensions)])**0.5
                    distances.append(distance)
            distances.sort()
            all_distances.append(distances)
            fitness[i] = raw_fitness[i] + 1/(distances[k_n] + 2)
        return fitness
    
    def trim_archive(self, archive, fitness):
        while len(archive) > self.archive_size:
            all_distances = []
            for i in range(len(archive)):
                distances = []
                for j in range(len(archive)):
                    if i != j:
                        sol_1 = archive[i]
                        sol_2 = archive[j]
                        distance = sum([(sol_1.objectives[i] - sol_2.objectives[i])**2 for i in range(self.objective_functions_dimensions)])
                        distances.append(distance)
                distances.sort()
                all_distances.append(distances)
            k_n = 1
            while True: 
                closest_n = min(all_distances[i][k_n] for i in range(len(all_distances)))
                most_crowded = [i for i in range(len(all_distances)) if all_distances[i][k_n] == closest_n]
                if len(most_crowded) == 1: 
                    archive.pop(most_crowded[0])
                    fitness.pop(most_crowded[0])
                    break
                else:
                    k_n = k_n + 1
        return archive, fitness
            
    def sort_population(self, population, fitness):
        sorted_ids = np.argsort(fitness)
        new_fitness = [fitness[sorted_ids[i]] for i in range(self.length)]
        new_population = [population[sorted_ids[i]] for i in range(self.length)]
        return new_population, new_fitness
    
    class roulette_wheel:
        def __init__(self, fitness, length):
            self.length = length
            self.roulette = np.zeros(self.length)
            for i in range(self.length):
                self.roulette[i] = 1 / (1 + abs(fitness[i]))
            total = sum(self.roulette)
            self.roulette[0] = self.roulette[0] / total
            for i in range(1, self.length):
                self.roulette[i] = self.roulette[i-1] +  self.roulette[i] / total
        
        def get_random_index(self):
            rand = int.from_bytes(os.urandom(8), byteorder="big") / ((1<<64) -1)
            for index in range(self.length):
                if rand <= self.roulette[index]:
                    return index
            return self.length - 1
    
    def breeding(self, population, fitness):
        offsprings = copy.deepcopy(population)
        b_off = 0
        roulette = self.roulette_wheel(fitness, self.archive_size)
        for i in range(self.population_size):
            # losujemy rodziców
            parent_1 = roulette.get_random_index()
            parent_2 = roulette.get_random_index()
            while parent_1 == parent_2:
                parent_2 = roulette.get_random_index()
            # krzyżowanie
            for j in range(self.variable_dimensions):
                rand = int.from_bytes(os.urandom(8), byteorder="big") / ((1<<64) -1)
                rand_b = int.from_bytes(os.urandom(8), byteorder="big") / ((1<<64) -1)
                if rand <= 0.5:
                    b_off = (2 * rand_b)**0.5
                else:
                    b_off = (1/(2 * (1 - rand_b)))**0.5
                offsprings[i].variables[j] = np.clip(
                        ((1 + b_off)*population[parent_1].variables[j] + (1 - b_off)*population[parent_2].variables[j])/2,
                        self.variable_bounds[j][0],
                        self.variable_bounds[j][1])
                if i < self.population_size - 1:
                    offsprings[i + 1].variables[j] = np.clip(
                            ((1 - b_off)*population[parent_1].variables[j] + (1 + b_off)*population[parent_2].variables[j])/2,
                            self.variable_bounds[j][0],
                            self.variable_bounds[j][1])
            offsprings[i].objectives = [objective_function(offsprings[i].variables) for objective_function in self.objective_functions]
        return offsprings
    
    def mutation(self, new_generation):
        d_mutation = 0
        for i in range(self.population_size):
            for j in range(self.variable_dimensions):
                prob = int.from_bytes(os.urandom(8), byteorder="big") / ((1<<64) -1)
                if prob < self.mutation_rate:
                    rand = int.from_bytes(os.urandom(8), byteorder="big") / ((1<<64) -1)
                    rand_d = int.from_bytes(os.urandom(8), byteorder="big") / ((1<<64) -1)
                    if rand <= 0.5:
                        d_mutation = (2 * rand_d)**(0.5) -1
                    else:
                        d_mutation = 1 - (2 * (1 - rand_d))**0.5
                    new_generation[i].variables[j] = np.clip(
                            new_generation[i].variables[j] + d_mutation,
                            self.variable_bounds[j][0],
                            self.variable_bounds[j][1])
            new_generation[i].objectives = [objective_function(new_generation[i].variables) for objective_function in self.objective_functions]
        return new_generation
    
    # główny algorytm
    def search(self):
        population = [self.solution(self.variable_bounds, self.objective_functions) for i in range(self.population_size)]
        archive = [self.solution(self.variable_bounds, self.objective_functions) for i in range(self.archive_size)]
        gen_id = 1
        print("Generation {}/{}".format(gen_id, self.generations))
        while True:
            if gen_id % 10 == 0:
                print("Generation {}/{}".format(gen_id, self.generations))
            population = population + archive
            raw_fitness = self.calculate_raw_fitness(population)
            fitness = self.calculate_fitness_and_distances(population, raw_fitness)
            population, fitness= self.sort_population(population, fitness)
            population = population[:self.population_size:]
            fitness = fitness[:self.population_size:]
            archive = [population[i] for i in range(self.population_size) if fitness[i] < 1]
            if len(archive) < self.archive_size:
                archive = archive + population[len(archive):self.archive_size]
            else:
                archive, fitness = self.trim_archive(archive, fitness)
            if gen_id >= self.generations:
                return archive
            population = self.breeding(population, fitness)
            population = self.mutation(population)
            gen_id = gen_id + 1
    
    def __init__(self,
                 population_size=5,
                 archive_size=None,
                 generations=50,
                 variable_bounds=[(-5,5)],
                 objective_functions=[None,None],
                 mutation_rate=0.1):
        population_size = abs(population_size)
        archive_size = int(population_size**0.5) if (archive_size is None or abs(archive_size) > population_size) else archive_size
        generations = abs(generations)
        self.population_size = population_size
        self.archive_size = archive_size
        self.generations = generations
        self.variable_dimensions = len(variable_bounds)
        self.variable_bounds = variable_bounds
        self.objective_functions = objective_functions
        self.objective_functions_dimensions = len(objective_functions)
        self.length = self.population_size + self.archive_size
        self.mutation_rate = mutation_rate
        
def create_plot(results):
    ax_variable_values= {}
    ax_objective_values= {}
    for i in range(len(results[0].variables)):
        ax_variable_values[i] = [x.variables[i] for x in results]
    for i in range(len(results[0].objectives)):
        ax_objective_values[i] = [x.objectives[i] for x in results]
    fig = plt.figure(figsize=(9,4))
    
    if len(ax_variable_values.keys()) == 1:
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_title("variables")
        ax1.scatter(ax_variable_values[0], np.zeros(len(ax_variable_values[0])), marker=".")
        ax1.set_yticklabels([])
        ax1.set_xlabel("x variable")
    elif len(ax_variable_values.keys()) == 2:
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_title("variables")
        ax1.scatter(ax_variable_values[0], ax_variable_values[1], marker=".")
        ax1.set_xlabel("x variable")
        ax1.set_ylabel("y variable")
    elif len(ax_variable_values.keys()) == 3:
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.set_title("variables")
        ax1.scatter(ax_variable_values[0], ax_variable_values[1], ax_variable_values[2], marker=".")
        ax1.set_xlabel("x variable")
        ax1.set_ylabel("y variable")
        ax1.set_zlabel("z variable")
    # fenotype plot:
    if len(ax_objective_values.keys()) == 1:
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_title("objective functions")
        ax2.scatter(ax_objective_values[0], np.zeros(len(ax_objective_values[0])), marker=".")
        ax2.set_yticklabels([])
        ax2.set_xlabel("function 1")
    elif len(ax_objective_values.keys()) == 2:
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_title("objective functions")
        ax2.scatter(ax_objective_values[0], ax_objective_values[1], marker=".")
        ax2.set_xlabel("function 1")
        ax2.set_ylabel("function 2")
    elif len(ax_objective_values.keys()) == 3:
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.set_title("objective functions")
        ax2.scatter(ax_objective_values[0], ax_objective_values[1], ax_objective_values[2], marker=".")
        ax2.set_xlabel("function 1")
        ax2.set_ylabel("function 2")
        ax2.set_zlabel("function 3")
    ax1.locator_params(nbins=5)
    ax2.locator_params(nbins=5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # MOP1:
    def MOP1_f1(variables=[0]):
        return variables[0]**2
    
    def MOP1_f2(variables=[0]):
        return (variables[0] - 2)**2
    
    MOP1_population_size = 100
    MOP1_archive_size = 20
    MOP1_generations_limit = 100
    MOP1_objective_functions = [MOP1_f1, MOP1_f2]
    MOP1_variable_bounds = [(- 10 ** 5, 10 ** 5)]
    
    # MOP2:
    def MOP2_f1(variables=[3]):
        return 1 - np.exp(-sum((x - 1/(len(variables)**0.5))**2 for x in variables))
    
    def MOP2_f2(variables=[3]):
        return 1 - np.exp(-sum((x + 1/(len(variables)**0.5))**2 for x in variables))
    
    MOP2_population_size = 200
    MOP2_archive_size = 40
    MOP2_generations_limit = 100
    MOP2_objective_functions = [MOP2_f1, MOP2_f2]
    MOP2_variable_bounds = [(-4, 4) for i in range(3)]
    
    def MOP3_f1(variables=[2]):
        A1 = 0.5*np.sin(1) - 2*np.cos(1) + np.sin(2) - 1.5*np.cos(2)
        A2 = 1.5*np.sin(1) - np.cos(1) + 2*np.sin(2) - 0.5*np.cos(2)
        B1 = 0.5*np.sin(variables[0]) - 2*np.cos(variables[0]) + np.sin(variables[1]) - 1.5*np.cos(variables[1])
        B2 = 1.5*np.sin(variables[0]) - np.cos(variables[0]) + 2*np.sin(variables[1]) - 0.5*np.cos(variables[1])
        return (1 + (A1 - B1)**2 + (A2 - B2)**2)
    
    def MOP3_f2(variables=[2]):
        return ((variables[0] + 3)**2 + (variables[1] + 1)**2)
    
    MOP3_population_size = 300
    MOP3_archive_size = 60
    MOP3_generations_limit = 100
    MOP3_objective_functions = [MOP3_f1, MOP3_f2]
    MOP3_variable_bounds = [(-np.pi, np.pi) for i in range(2)]
    
    # MOP4
    def MOP4_f1(variables=[3]):
        return sum([-10 * np.e**((-0.2)* (variables[i]**2 + variables[i+1]**2)**0.5) for i in range(0, len(variables)-1)])
    
    def MOP4_f2(variables=[3]):
        return sum([abs(variable)**0.8 + 5*np.sin(variable)**3 for variable in variables])
    
    MOP4_population_size = 300
    MOP4_archive_size = 60
    MOP4_generations_limit = 100
    MOP4_objective_functions = [MOP4_f1, MOP4_f2]
    MOP4_variable_bounds = [(-5, 5) for i in range(3)]
    
    # MOP5
    def MOP5_f1(variables=[2]):
        return 0.5 * (variables[0]**2 + variables[1]**2) + np.sin(variables[0]**2 + variables[1]**2)
    
    def MOP5_f2(variables=[2]):
        return ((3*variables[0] - 2*variables[1] + 4)**2)/8 + ((variables[0] - variables[1] +1)**2)/27 + 15
    
    def MOP5_f3(variables=[2]):
        return 1/(variables[0]**2 + variables[1]**2 + 1) - 1.1*np.e**(-variables[0]**2 - variables[1]**2)
    
    MOP5_population_size = 300
    MOP5_archive_size = 60
    MOP5_generations_limit = 5
    MOP5_objective_functions = [MOP5_f1, MOP5_f2, MOP5_f3]
    MOP5_variable_bounds = [(-30, 30) for i in range(2)]
    
    # MOP6
    def MOP6_f1(variables=[3]):
        return variables[0]
    
    def MOP6_f2(variables=[3]):
        return (1 + 10*variables[1]) * (1 - (variables[0] / (1 + 10*variables[1]))**2 - (variables[0] / (1 + 10*variables[1]))*np.sin(8*np.pi*variables[0]))
    
    MOP6_population_size = 300
    MOP6_archive_size = 100
    MOP6_generations_limit = 100
    MOP6_objective_functions = [MOP6_f1, MOP6_f2]
    MOP6_variable_bounds = [(0, 1) for i in range(2)]
    
    # MOP7
    def MOP7_f1(variables=[2]):
        return ((variables[0] - 2)**2)/2 + ((variables[1] + 1)**2)/13 + 3
    
    def MOP7_f2(variables=[2]):
        return ((variables[0] + variables[1] - 3)**2)/36 + ((variables[1] - variables[0] + 2)**2)/8 - 17
    
    def MOP7_f3(variables=[2]):
        return ((variables[0] + 2*variables[1] - 1)**2)/175 + ((2*variables[1] - variables[0])**2)/17 - 13
    
    MOP7_population_size = 300
    MOP7_archive_size = 100
    MOP7_generations_limit = 100
    MOP7_objective_functions = [MOP7_f1, MOP7_f2, MOP7_f3]
    MOP7_variable_bounds = [(-400, 400) for i in range(2)]
    
    
    # rozwiązania
    #"""
    print("MOP1:")
    test = SPEA2(MOP1_population_size,
                 MOP1_archive_size,
                 MOP1_generations_limit,
                 MOP1_variable_bounds,
                 MOP1_objective_functions)
    results = test.search()
    print("Found {} solutions".format(len(results)))
    create_plot(results)
    #"""
    """
    print("MOP2:")
    test = SPEA2(MOP2_population_size,
                 MOP2_archive_size,
                 MOP2_generations_limit,
                 MOP2_variable_bounds,
                 MOP2_objective_functions)
    results = test.search()
    print("Found {} solutions".format(len(results)))
    create_plot(results)
    """
    """
    print("MOP3:")
    test = SPEA2(MOP3_population_size,
                 MOP3_archive_size,
                 MOP3_generations_limit,
                 MOP3_variable_bounds,
                 MOP3_objective_functions)
    results = test.search()
    print("Found {} solutions".format(len(results)))
    create_plot(results)
    """
    """
    print("MOP4:")
    test = SPEA2(MOP4_population_size,
                 MOP4_archive_size,
                 MOP4_generations_limit,
                 MOP4_variable_bounds,
                 MOP4_objective_functions)
    results = test.search()
    print("Found {} solutions".format(len(results)))
    create_plot(results)
    """
    """
    print("MOP5:")
    test = SPEA2(MOP5_population_size,
                 MOP5_archive_size,
                 MOP5_generations_limit,
                 MOP5_variable_bounds,
                 MOP5_objective_functions)
    results = test.search()
    print("Found {} solutions".format(len(results)))
    create_plot(results)
    """
    """
    print("MOP6:")
    test = SPEA2(MOP6_population_size,
                 MOP6_archive_size,
                 MOP6_generations_limit,
                 MOP6_variable_bounds,
                 MOP6_objective_functions)
    results = test.search()
    print("Found {} solutions".format(len(results)))
    create_plot(results)
    """
    """
    print("MOP7:")
    test = SPEA2(MOP7_population_size,
                 MOP7_archive_size,
                 MOP7_generations_limit,
                 MOP7_variable_bounds,
                 MOP7_objective_functions)
    results = test.search()
    print("Found {} solutions".format(len(results)))
    create_plot(results)
    """
    