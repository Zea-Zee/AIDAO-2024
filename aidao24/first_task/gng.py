import numpy as np
import random

class GNGNode:
    def __init__(self, position):
        self.position = np.array(position)
        self.error = 0
        self.neighbors = set()

class GrowingNeuralGas:
    def __init__(self, input_dim, max_nodes=100, max_age=50, epsilon_b=0.05, epsilon_n=0.006, alpha=0.5, beta=0.0005, lambda_=100):
        self.input_dim = input_dim
        self.max_nodes = max_nodes
        self.max_age = max_age
        self.epsilon_b = epsilon_b
        self.epsilon_n = epsilon_n
        self.alpha = alpha
        self.beta = beta
        self.lambda_ = lambda_
        self.nodes = []
        self.edges = {}
        self.iteration = 0

    def initialize(self, data):
        idx = random.sample(range(len(data)), 2)
        self.nodes.append(GNGNode(data[idx[0]]))
        self.nodes.append(GNGNode(data[idx[1]]))

    def find_nearest_nodes(self, point):
        distances = [np.linalg.norm(node.position - point) for node in self.nodes]
        sorted_indices = np.argsort(distances)
        return sorted_indices[0], sorted_indices[1]

    def update_node_positions(self, s, t, point):
        s.position += self.epsilon_b * (point - s.position)
        for neighbor in s.neighbors:
            neighbor.position += self.epsilon_n * (point - neighbor.position)

    def increment_edge_ages(self, s):
        for neighbor in s.neighbors:
            if (s, neighbor) in self.edges:
                self.edges[(s, neighbor)] += 1
            elif (neighbor, s) in self.edges:  # Проверяем, что ребро может быть в обратном порядке
                self.edges[(neighbor, s)] += 1

    def add_edge(self, s, t):
        s.neighbors.add(t)
        t.neighbors.add(s)
        self.edges[(s, t)] = 0

    def remove_old_edges(self):
        edges_to_remove = [edge for edge, age in self.edges.items() if age > self.max_age]
        for edge in edges_to_remove:
            s, t = edge
            s.neighbors.remove(t)
            t.neighbors.remove(s)
            del self.edges[edge]

    def insert_new_node(self):
        # Find the node with the largest accumulated error
        q = max(self.nodes, key=lambda node: node.error)
        # Find the neighbor of q with the largest accumulated error
        f = max(q.neighbors, key=lambda node: node.error)
        n = GNGNode((q.position + f.position) / 2)
        self.nodes.append(n)
        # Adjust errors
        q.error *= self.alpha
        f.error *= self.alpha
        n.error = q.error
        # Adjust edges
        q.neighbors.remove(f)
        f.neighbors.remove(q)
        self.add_edge(q, n)
        self.add_edge(f, n)

    def update(self, data):
        self.iteration += 1

        # Randomly select a data point
        point = random.choice(data)

        # Find two nearest nodes
        s_idx, t_idx = self.find_nearest_nodes(point)
        s = self.nodes[s_idx]
        t = self.nodes[t_idx]

        # Update error of the nearest node
        s.error += np.linalg.norm(s.position - point) ** 2

        # Move the nearest node and its neighbors
        self.update_node_positions(s, t, point)

        # Increment the age of edges connected to s
        self.increment_edge_ages(s)

        # Add edge between s and t
        self.add_edge(s, t)

        # Remove edges older than max_age
        self.remove_old_edges()

        # Insert new node every λ iterations
        if self.iteration % self.lambda_ == 0 and len(self.nodes) < self.max_nodes:
            self.insert_new_node()

        # Decrease errors over time
        for node in self.nodes:
            node.error *= (1 - self.beta)

    def train(self, data, iterations=1000):
        self.initialize(data)
        for _ in range(iterations):
            self.update(data)

