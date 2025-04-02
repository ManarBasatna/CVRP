import random
import numpy as np
import csv
from datetime import datetime
from cvrp import CVRP

class RandomAlgorithm:
    def __init__(self, cvrp, num_iterations=1000):
        self.cvrp = cvrp
        self.num_iterations = num_iterations
        self.stats = {
            'best': [],
            'average': [],
            'worst': []
        }

    def run(self):
        best_solution = None
        best_cost = float('inf')
        costs = []
        
        for _ in range(self.num_iterations):
            solution = self._random_solution()
            cost = self._calculate_cost(solution)
            costs.append(cost)
            
            if cost < best_cost:
                best_cost = cost
                best_solution = solution
                
            self._update_stats(costs)
        
        return self._prepare_results(best_solution, costs)

    def _random_solution(self):
        customers = list(range(1, len(self.cvrp.locations)))
        random.shuffle(customers)
        return self._split_to_routes(customers)

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

    def _update_stats(self, costs):
        self.stats['best'].append(min(costs))
        self.stats['average'].append(np.mean(costs))
        self.stats['worst'].append(max(costs))

    def _prepare_results(self, best_solution, costs):
        return {
            'best_solution': best_solution,
            'best_distance': min(costs),
            'average_distance': np.mean(costs),
            'worst_distance': max(costs),
            'std_dev': np.std(costs),
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