import random
import numpy as np
import csv
import os
from datetime import datetime
from cvrp import CVRP

class GreedyAlgorithm:
    def __init__(self, cvrp, num_runs=10):
        self.cvrp = cvrp
        self.num_runs = num_runs  # Number of runs for statistics
        self.stats = {
            'best': [],
            'average': [],
            'worst': []
        }

    def run(self):
        solutions = []
        costs = []
        
        for _ in range(self.num_runs):
            solution = self._greedy_solution()
            cost = self._calculate_cost(solution)
            solutions.append(solution)
            costs.append(cost)
            self._update_stats(costs)
        
        best_idx = np.argmin(costs)
        return self._prepare_results(solutions[best_idx], costs)

    def _greedy_solution(self):
        """Generates a solution using greedy nearest neighbor approach"""
        # Create a copy of customers to avoid modifying the original list
        customers = list(range(1, len(self.cvrp.locations)))
        random.shuffle(customers)  # Random starting point for variation
        
        routes = []
        current_route = [0]  # Start at depot
        current_load = 0
        
        while customers:
            last_city = current_route[-1]
            
            # Find reachable cities sorted by distance
            candidates = [
                c for c in customers 
                if self.cvrp.demands[c] + current_load <= self.cvrp.vehicle_capacity
            ]
            
            if not candidates:
                # Return to depot and start new route
                current_route.append(0)
                routes.append(current_route)
                current_route = [0]
                current_load = 0
                continue
                
            # Select nearest feasible city
            next_city = min(
                candidates,
                key=lambda c: self.cvrp.distance_matrix[last_city][c]
            )
            
            current_route.append(next_city)
            current_load += self.cvrp.demands[next_city]
            customers.remove(next_city)
        
        # Complete final route if it has customers
        if len(current_route) > 1:
            current_route.append(0)
            routes.append(current_route)
            
        return routes

    def _calculate_cost(self, solution):
        """Calculates total distance for a solution"""
        if not solution:
            return float('inf')
        return sum(self.cvrp.calculate_route_distance(route) for route in solution)

    def _update_stats(self, costs):
        """Updates running statistics"""
        if costs:  # Only update if we have costs
            self.stats['best'].append(min(costs))
            self.stats['average'].append(np.mean(costs))
            self.stats['worst'].append(max(costs))

    def _prepare_results(self, best_solution, costs):
        """Prepares final results dictionary"""
        if not costs:  # Handle empty case
            return {
                'best_solution': [],
                'best_distance': float('inf'),
                'average_distance': 0,
                'worst_distance': 0,
                'std_dev': 0,
                'stats': self.stats
            }
            
        return {
            'best_solution': best_solution,
            'best_distance': min(costs),
            'average_distance': np.mean(costs),
            'worst_distance': max(costs),
            'std_dev': np.std(costs) if len(costs) > 1 else 0.0,
            'stats': self.stats
        }

    @staticmethod
    def log_to_csv(results, instance_name, algorithm_name):
        """Logs """
        filename = f"cvrp_results_{datetime.now().strftime('%Y%m%d')}.csv"
        file_exists = os.path.isfile(filename)
        
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    'Timestamp', 'Instance', 'Algorithm',
                    'Best', 'Average', 'Worst', 'StdDev', 'NumRuns'
                ])
                
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                instance_name,
                algorithm_name,
                f"{results['best_distance']:.2f}",
                f"{results['average_distance']:.2f}",
                f"{results['worst_distance']:.2f}",
                f"{results['std_dev']:.2f}",
                len(results['stats']['best'])
            ])