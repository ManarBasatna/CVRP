# utils.py
import math
from cvrp import CVRP

def read_vrp_file(file_path):
    """Reads standard VRP file format and returns CVRP instance"""
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    data = {
        'nodes': [],
        'demands': [],
        'depot_index': 0
    }
    current_section = None
    
    for line in lines:
        if line.startswith('NODE_COORD_SECTION'):
            current_section = 'NODE_COORD'
            continue
        elif line.startswith('DEMAND_SECTION'):
            current_section = 'DEMAND'
            continue
        elif line.startswith('DEPOT_SECTION'):
            current_section = 'DEPOT'
            continue
        elif line == 'EOF':
            break

        if current_section == 'NODE_COORD':
            parts = line.split()
            node_id = int(parts[0]) - 1  # Convert to 0-based index
            x = float(parts[1])
            y = float(parts[2])
            data['nodes'].append((node_id, x, y))
        elif current_section == 'DEMAND':
            parts = line.split()
            node_id = int(parts[0]) - 1
            demand = int(parts[1])
            data['demands'].append((node_id, demand))
        elif current_section == 'DEPOT':
            if line.strip() != '-1':
                data['depot_index'] = int(line.strip()) - 1
        elif ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().upper()
            data[key] = value.strip()

    # Process nodes and demands
    n = len(data['nodes'])
    coords = [None] * n
    demands = [0] * n
    
    for node_id, x, y in data['nodes']:
        coords[node_id] = (x, y)
    
    for node_id, demand in data['demands']:
        demands[node_id] = demand

    # Create distance matrix
    distances = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = coords[i]
            x2, y2 = coords[j]
            dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            distances[i][j] = dist
            distances[j][i] = dist

    # Separate depot and customers
    depot = coords[data['depot_index']]
    customer_coords = [coord for i, coord in enumerate(coords) if i != data['depot_index']]
    customer_demands = [demand for i, demand in enumerate(demands) if i != data['depot_index']]

    return CVRP(
        depot=depot,
        locations=customer_coords,
        demands=customer_demands,
        vehicle_capacity=int(data.get('CAPACITY', 100)),  
        distances=distances
    )