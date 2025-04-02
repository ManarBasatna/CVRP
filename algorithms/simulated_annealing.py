import random
import math
import numpy as np
import csv
from datetime import datetime
from cvrp import CVRP

class SimulatedAnnealing:
    def __init__(self, cvrp, initial_temp=1000, cooling_rate=0.99, iterations=1000):
        self.cvrp = cvrp
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.iterations = iterations
        self.stats = {
            'best': [],
            'current': [],
            'worst': []
        }

    def run(self):
        current_solution = self._create_initial_solution()
        current_cost = self._calculate_cost(current_solution)
        best_solution = current_solution.copy()
        best_cost = current_cost
        temp = self.initial_temp
        
        for i in range(self.iterations):
            new_solution = self._get_neighbor(current_solution)
            new_cost = self._calculate_cost(new_solution)
            
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_solution = new_solution
                current_cost = new_cost
                
                if new_cost < best_cost:
                    best_solution = new_solution
                    best_cost = new_cost
            
            temp *= self.cooling_rate
            self._update_stats(i, best_cost, current_cost)
        
        return self._prepare_results(best_solution, best_cost)

    def _create_initial_solution(self):
        customers = list(range(1, len(self.cvrp.locations)))
        random.shuffle(customers)
        return self._split_to_routes(customers)

    def _get_neighbor(self, solution):
        flat = [c for route in solution for c in route[1:-1]]
        if len(flat) < 2:
            return solution
            
        i, j = random.sample(range(len(flat)), 2)
        new_flat = flat.copy()
        new_flat[i], new_flat[j] = new_flat[j], new_flat[i]
        return self._split_to_routes(new_flat)

    def _split_to_routes(self, customers):
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

    def _calculate_cost(self, solution):
        return sum(self.cvrp.calculate_route_distance(route) for route in solution)

    def _update_stats(self, iteration, best_cost, current_cost):
        self.stats['best'].append(best_cost)
        self.stats['current'].append(current_cost)
        self.stats['worst'].append(max(self.stats['best'] + [current_cost]))

    def _prepare_results(self, best_solution, best_cost):
        return {
            'best_solution': best_solution,
            'best_distance': best_cost,
            'average_distance': np.mean(self.stats['current']),
            'worst_distance': max(self.stats['worst']),
            'std_dev': np.std(self.stats['current']),
            'stats': self.stats
        }

    @staticmethod
    def log_to_csv(results, instance_name, algorithm_name):
        filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                instance_name,
                algorithm_name,
                results['best_distance'],
                results['average_distance'],
                results['worst_distance'],
                results['std_dev'],
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ])