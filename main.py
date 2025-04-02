from cvrp import CVRP, load_vrp_file
from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.random_algorithm import RandomAlgorithm
from algorithms.greedy_algorithm import GreedyAlgorithm
from algorithms.tabu_search import TabuSearch
from algorithms.simulated_annealing import SimulatedAnnealing
import os
import time
import csv
from datetime import datetime

def print_statistics(results: dict, instance: str, algorithm: str, exec_time: float):
    """Print formatted results for any algorithm"""
    print(f"\n{'='*60}")
    print(f"{instance} - {algorithm.upper()} RESULTS")
    print(f"{'='*60}")
    print(f"⏱ Computation time: {exec_time:.2f}s")
    print(f"🏆 Best distance: {results['best_distance']:.2f}")
    print(f"📈 Average distance: {results['average_distance']:.2f}")
    print(f"🔻 Worst distance: {results['worst_distance']:.2f}")
    print(f"📉 Standard deviation: {results['std_dev']:.2f}")
    
    print("\n🚛 BEST SOLUTION ROUTES:")
    for i, route in enumerate(results['best_solution'], 1):
        demand = sum(results['cvrp'].demands[node] for node in route)
        distance = results['cvrp'].calculate_route_distance(route)
        print(f"Route {i}: {' → '.join(map(str, route))}")
        print(f"   Demand: {demand}/{results['cvrp'].vehicle_capacity} | Distance: {distance:.2f}")

def log_to_csv(results: dict, instance: str, algorithm: str):
    """ CSV logging function"""
    filename = f"cvrp_results_{datetime.now().strftime('%Y%m%d')}.csv"
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                'Timestamp', 'Instance', 'Algorithm',
                'Best', 'Average', 'Worst', 'StdDev'
            ])
            
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            instance,
            algorithm,
            f"{results['best_distance']:.2f}",
            f"{results['average_distance']:.2f}",
            f"{results['worst_distance']:.2f}",
            f"{results['std_dev']:.2f}"
        ])

def process_algorithm(cvrp_instance, instance_name, algorithm_class, config):
    """Run an algorithm and handle results"""
    try:
        start_time = time.time()
        algorithm = algorithm_class(cvrp_instance, **config)
        results = algorithm.run()
        results['cvrp'] = cvrp_instance  # Add problem reference
        exec_time = time.time() - start_time
        
        print_statistics(results, instance_name, algorithm_class.__name__, exec_time)
        log_to_csv(results, instance_name, algorithm_class.__name__)
        
    except Exception as e:
        print(f"\n❌ Error in {algorithm_class.__name__}: {str(e)}")
        return None

def main():
    print("=== CVRP SOLVER SUITE ===")
    print("Algorithms: GA, Random, Greedy, Tabu Search, Simulated Annealing\n")
    
    # Configuration
    instance_folder = "instances"
    instances = [f"{i}.vrp" for i in range(1, 8)]  # 1.vrp to 7.vrp, i renamed them like that
    
    # Algorithm configurations
    algorithms = {
        GeneticAlgorithm: {
            'pop_size': 100,
            'generations': 200,
            'mutation_rate': 0.15,
            'crossover_rate': 0.85,
            'tournament_size': 5
        },
        RandomAlgorithm: {
            'num_iterations': 1000
        },
        GreedyAlgorithm: {},  
        TabuSearch: {
            'iterations': 500,
            'tabu_size': 50,
            'neighborhood_size': 20
        },
        SimulatedAnnealing: {
            'initial_temp': 1000,
            'cooling_rate': 0.99,
            'iterations': 1000
        }
    }

    # Validate instances folder
    if not os.path.exists(instance_folder):
        print(f"\n❌ Error: Create '{instance_folder}' folder with .vrp files!")
        return

    # Process each instance
    for instance in instances:
        file_path = os.path.join(instance_folder, instance)
        if not os.path.exists(file_path):
            print(f"\n⚠ Missing: {instance} - Skipping")
            continue
            
        try:
            print(f"\n{'#'*60}")
            print(f"PROCESSING INSTANCE: {instance}")
            print(f"{'#'*60}")
            
            # Load problem instance
            problem = load_vrp_file(file_path)
            
            # Run all algorithms
            for algorithm_class, config in algorithms.items():
                process_algorithm(problem, instance, algorithm_class, config)
                
        except Exception as e:
            print(f"\n❌ Critical error processing {instance}: {str(e)}")
            continue

if __name__ == "__main__":
    main()