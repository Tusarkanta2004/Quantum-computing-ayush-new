import numpy as np
import networkx as nx
from itertools import product
import random

class ToricCode:
    def __init__(self, L):
        self.L = L
        self.n = L*L
        # Qubits: represented as a 2D lattice LxL with two types (horizontal and vertical)
        self.qubits = np.zeros((L, L, 2), dtype=int)  # 0=no error, 1=X,2=Z,3=Y
        self.syndrome_X = np.zeros((L, L), dtype=int)
        self.syndrome_Z = np.zeros((L, L), dtype=int)

    def apply_random_errors(self, p):
        """Apply random Pauli X, Y, Z errors with probability p."""
        for x in range(self.L):
            for y in range(self.L):
                for t in range(2):  # two qubits per site (horizontal=0, vertical=1)
                    r = random.random()
                    if r < p:
                        self.qubits[x, y, t] = random.choice([1, 2, 3])
    
    def measure_syndrome(self):
        """Measure X- and Z- syndromes for all stabilizers."""
        L = self.L
        self.syndrome_X = np.zeros((L,L), dtype=int)
        self.syndrome_Z = np.zeros((L,L), dtype=int)

        # Vertex (star) operators: measure X-type
        for x in range(L):
            for y in range(L):
                # sum of Z-errors on edges connected to vertex
                val = 0
                for dx, dy, t in [(0,0,0),(0,0,1),(L-1,0,0),(0,L-1,1)]:
                    nx = (x+dx)%L
                    ny = (y+dy)%L
                    if self.qubits[nx, ny, t] in [2,3]:  # Z or Y error
                        val ^= 1
                self.syndrome_X[x,y] = val
        
        # Plaquette operators: measure Z-type
        for x in range(L):
            for y in range(L):
                val = 0
                for dx, dy, t in [(0,0,0),(0,0,1),(L-1,0,1),(0,L-1,0)]:
                    nx = (x+dx)%L
                    ny = (y+dy)%L
                    if self.qubits[nx, ny, t] in [1,3]:  # X or Y error
                        val ^= 1
                self.syndrome_Z[x,y] = val

    def mwpm_decode(self, syndrome):
        """Very simple MWPM decoder using networkx shortest paths."""
        # Extract non-zero syndrome positions
        defects = [(x,y) for x in range(self.L) for y in range(self.L) if syndrome[x,y]==1]
        if len(defects)%2 != 0:
            print("Odd number of defects, cannot decode perfectly.")
            return

        G = nx.Graph()
        for i, a in enumerate(defects):
            G.add_node(i, pos=a)
        
        # Complete graph with distances
        for i, a in enumerate(defects):
            for j, b in enumerate(defects):
                if i < j:
                    dist = (abs(a[0]-b[0]) + abs(a[1]-b[1]))  # Manhattan distance
                    G.add_edge(i,j,weight=dist)

        # Find MWPM
        matching = nx.algorithms.matching.min_weight_matching(G, maxcardinality=True)
        # Apply corrections along shortest paths
        for i,j in matching:
            a = G.nodes[i]['pos']
            b = G.nodes[j]['pos']
            # Simple straight line path (does not wrap torus perfectly)
            x1,y1 = a
            x2,y2 = b
            for x in range(min(x1,x2), max(x1,x2)+1):
                for y in range(min(y1,y2), max(y1,y2)+1):
                    self.qubits[x%self.L, y%self.L, 0] ^= 1  # apply X correction as example

    def logical_error_check(self):
        """Check if logical X or Z errors exist."""
        L = self.L
        # Horizontal logical operator
        logX = np.prod([self.qubits[x,0,0] for x in range(L)]) %2
        # Vertical logical operator
        logZ = np.prod([self.qubits[0,y,1] for y in range(L)]) %2
        return logX, logZ


# Example usage
L = 5
p_error = 0.1
code = ToricCode(L)
code.apply_random_errors(p_error)
code.measure_syndrome()
code.mwpm_decode(code.syndrome_X)
logX, logZ = code.logical_error_check()
print("Logical X error:", logX, "Logical Z error:", logZ)
