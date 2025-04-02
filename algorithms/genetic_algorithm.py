import random
import numpy as np
from typing import List, Dict
from cvrp import CVRP

class GeneticAlgorithm:
    def __init__(self, cvrp: CVRP, pop_size: int = 100, generations: int = 500,
                 mutation_rate: float = 0.1, crossover_rate: float = 0.8,
                 tournament_size: int = 5, elitism: float = 0.1):
        self.cvrp = cvrp
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism = int(elitism * pop_size)
        self.population = self._initialize_population()
        
    def _initialize_population(self) -> List[List[List[int]]]:
        population = []
        while len(population) < self.pop_size:
            solution = self._generate_random_solution()
            if self.cvrp.validate_solution(solution):
                population.append(solution)
        return population
    
    def _generate_random_solution(self) -> List[List[int]]:
        customers = list(range(1, len(self.cvrp.locations)))
        random.shuffle(customers)
        routes = []
        current_route = [0]
        current_load = 0
        
        for customer in customers:
            demand = self.cvrp.demands[customer]
            if current_load + demand > self.cvrp.vehicle_capacity:
                current_route.append(0)
                routes.append(current_route)
                current_route = [0, customer]
                current_load = demand
            else:
                current_route.append(customer)
                current_load += demand
        
        if len(current_route) > 1:
            current_route.append(0)
            routes.append(current_route)
        return routes
    
    def _calculate_fitness(self, solution: List[List[int]]) -> float:
        return sum(self.cvrp.calculate_route_distance(route) for route in solution)
    
    def _tournament_selection(self) -> List[List[int]]:
        tournament = random.sample(self.population, self.tournament_size)
        return min(tournament, key=lambda x: self._calculate_fitness(x))
    
    def _ordered_crossover(self, parent1: List[List[int]], parent2: List[List[int]]) -> List[List[int]]:
        flat1 = [node for route in parent1 for node in route if node != 0][:-1]
        flat2 = [node for route in parent2 for node in route if node != 0][:-1]
        
        size = len(flat1)
        start, end = sorted(random.sample(range(size), 2))
        child = [-1] * size
        
        child[start:end] = flat1[start:end]
        
        ptr = 0
        for i in range(size):
            if ptr == start:
                ptr = end
            if ptr >= size:
                break
            if flat2[i] not in child[start:end]:
                child[ptr] = flat2[i]
                ptr += 1
                
        return self._split_to_routes(child)
    
    def _split_to_routes(self, customers: List[int]) -> List[List[int]]:
        routes = []
        current_route = [0]
        current_load = 0
        
        for customer in customers:
            demand = self.cvrp.demands[customer]
            if current_load + demand > self.cvrp.vehicle_capacity:
                current_route.append(0)
                routes.append(current_route)
                current_route = [0, customer]
                current_load = demand
            else:
                current_route.append(customer)
                current_load += demand
        
        if len(current_route) > 1:
            current_route.append(0)
            routes.append(current_route)
        return routes
    
    def _mutate(self, solution: List[List[int]]) -> List[List[int]]:
        if random.random() < self.mutation_rate:
            flat = [node for route in solution for node in route if node != 0]
            idx1, idx2 = random.sample(range(len(flat)), 2)
            flat[idx1], flat[idx2] = flat[idx2], flat[idx1]
            return self._split_to_routes(flat)
        return solution
    
    def run(self) -> Dict[str, any]:
        stats = {
            'best': [],
            'avg': [],
            'worst': []
        }
        
        for gen in range(self.generations):
            # Evaluate population
            fitness = [self._calculate_fitness(ind) for ind in self.population]
            
            # Record stats
            stats['best'].append(min(fitness))
            stats['avg'].append(np.mean(fitness))
            stats['worst'].append(max(fitness))
            
            # Sort population
            self.population.sort(key=lambda x: self._calculate_fitness(x))
            
            # Keep elites
            new_pop = self.population[:self.elitism]
            
            # Generate offspring
            while len(new_pop) < self.pop_size:
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                
                if random.random() < self.crossover_rate:
                    child = self._ordered_crossover(parent1, parent2)
                else:
                    child = [route.copy() for route in parent1]
                
                child = self._mutate(child)
                if self.cvrp.validate_solution(child):
                    new_pop.append(child)
            
            self.population = new_pop
        
        # Final evaluation
        final_fitness = [self._calculate_fitness(ind) for ind in self.population]
        
        return {
            'best_solution': min(self.population, key=lambda x: self._calculate_fitness(x)),
            'best_distance': min(final_fitness),
            'average_distance': np.mean(final_fitness),
            'worst_distance': max(final_fitness),
            'std_dev': np.std(final_fitness),
            'stats': stats  # Generational statistics
        }