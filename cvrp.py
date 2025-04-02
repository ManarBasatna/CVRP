import math
from typing import List, Tuple, Dict
import os

class CVRP:
    def __init__(self, depot: Tuple[float, float], locations: List[Tuple[float, float]], 
                 demands: List[int], vehicle_capacity: int):
        self.depot = depot
        self.locations = [depot] + locations  # depot is index 0
        self.demands = [0] + demands  # depot has 0 demand
        self.vehicle_capacity = vehicle_capacity
        self.distance_matrix = self._create_distance_matrix()
        
    def _create_distance_matrix(self) -> List[List[float]]:
        """Create symmetric distance matrix between all locations"""
        n = len(self.locations)
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = self.locations[i]
                x2, y2 = self.locations[j]
                dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                matrix[i][j] = dist
                matrix[j][i] = dist
        return matrix

    def calculate_route_distance(self, route: List[int]) -> float:
        """Calculate total distance for a given route"""
        distance = 0.0
        for i in range(len(route)-1):
            from_node = route[i]
            to_node = route[i+1]
            distance += self.distance_matrix[from_node][to_node]
        return distance

    def validate_solution(self, routes: List[List[int]]) -> bool:
        """Check if solution meets all constraints"""
        # Check all customers visited exactly once (except depot)
        visited = set()
        for route in routes:
            for node in route[1:-1]:  # exclude depot nodes
                if node in visited or node <= 0 or node >= len(self.locations):
                    return False
                visited.add(node)
        
        # Check all customers visited
        if len(visited) != len(self.locations)-1:
            return False
            
        # Check capacity constraints
        for route in routes:
            route_demand = sum(self.demands[node] for node in route)
            if route_demand > self.vehicle_capacity:
                return False
                
        # Check routes start/end at depot
        for route in routes:
            if route[0] != 0 or route[-1] != 0:
                return False
                
        return True

def load_vrp_file(file_path: str) -> CVRP:
    """Load standard VRP file format"""
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    metadata = {}
    node_coords = []
    demands = []
    depot_index = 0
    
    section = None
    for line in lines:
        if line.startswith('NODE_COORD_SECTION'):
            section = 'COORD'
            continue
        elif line.startswith('DEMAND_SECTION'):
            section = 'DEMAND'
            continue
        elif line.startswith('DEPOT_SECTION'):
            section = 'DEPOT'
            continue
        elif line == 'EOF':
            break
            
        if section == 'COORD':
            parts = line.split()
            if len(parts) >= 3:
                node_coords.append((float(parts[1]), float(parts[2])))
        elif section == 'DEMAND':
            parts = line.split()
            if len(parts) >= 2:
                demands.append(int(parts[1]))
        elif section == 'DEPOT':
            if line.strip() != '-1':
                try:
                    depot_index = int(line.strip()) - 1
                except ValueError:
                    continue
        elif ':' in line:
            key, value = line.split(':', 1)
            metadata[key.strip()] = value.strip()
    
    if not node_coords or not demands:
        raise ValueError("Invalid VRP file - missing coordinates or demands")
    
    depot = node_coords[depot_index]
    locations = [coord for i, coord in enumerate(node_coords) if i != depot_index]
    demands = [demand for i, demand in enumerate(demands) if i != depot_index]
    
    return CVRP(
        depot=depot,
        locations=locations,
        demands=demands,
        vehicle_capacity=int(metadata.get('CAPACITY', 100))
    )