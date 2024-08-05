# FIT2004 ASSIGNMENT 2 WRITTEN BY TANG WEI YAO
from collections import deque
# QUESTION 1: Open reading frames

class Node:
    """
    Node data structure for the SuffixTrie.
    """
    def __init__(self, data=None):
        """
        Function Description:
            Constructor to initialise a new node

        Input:
            data: Optional parameter to store in the node

        Time complexity:
            Time complexity analysis: O(1), initialisation

        Space complexity:
            Input Space analysis: O(1) for the input
            Aux space analysis: O(1), O(4+1) for the link array which stores the 4 uppercase letters A-D and $ terminal node
        """
        self.link = [None] * (4 + 1)  # Uppercase letter A-D + $ for terminal
        self.data = data
        self.indexes = []  # List to store starting indexes of suffixes

class OrfFinder:
    """
    Class of Suffix Trie data structure utilised for finding substrings in a genome sequence.
    Reference used: [FIT2004_2020sem01] Lecture11 Trie, Week 11 Pass session
    """
    def __init__(self, genome):
        """
        Function description:
            Initializes the OrfFinder with the given genome string and constructs a suffix trie.

        Approach description:
            Initializes the genome string and constructs a suffix trie for efficient searching of substrings.
            The suffix trie is built by the SuffixTrie class, which inserts all suffixes of the genome into the trie.
            The letters are stored at the nodes of the tries instead of the edge.

        Input:
            genome: The genome string.

        Time complexity:
            Time complexity analysis: O(N^2), where N is the length of the genome.
        Space complexity:
            Input space analysis: O(N), where N is the length of the genome.
            Aux space analysis: O(N^2) for all the characters in the suffix trie nodes.
        """
        self.root = Node()  # Root node of the trie
        self.genome = genome
        for i in range(len(self.genome)):  # construct the suffix trie by inserting all suffixes starting from each index in the text
            self.insert(i)


    def insert(self, start_index):
        """
        Function description:
            Inserts a suffix into the trie starting from the given index.

        Approach description:
            Starting from the given index in the input text, iterates through each character of the suffix.
            For each character, calculates the appropriate index in the link array and checks if a node exists at that index.
            If not, creates a new node and inserts it into the link array. Updates the current node to the newly created node
            or existing node and appends the starting index to the node's indexes list.

        Input:
            start_index: The starting index of the suffix in the original text.

        Time complexity:
            Time complexity analysis: O(N), where N is the length of the suffix.
        Space complexity:
            Input space analysis: O(1) for the starting index.
            Aux space analysis: O(1) for node updates.
        """
        current = self.root  # Start at the root
        n = len(self.genome)
        for i in range(start_index, n):  # Iterate through the suffix characters
            index = ord(self.genome[i]) - ord('A') + 1  # Calculate the index for the character
            if current.link[index] is None:  # If no node exists at this index, create a new node
                current.link[index] = Node()
            current = current.link[index]  # Move to the next node
            current.indexes.append(start_index)  # Append the starting index of the suffix

    def find_sequence(self, sequence):
        """
        Function description:
            Finds all index of a sequence in the trie.

        Approach description:
            Traverses the trie using the characters of the substring sequence to discover the corresponding node.
            Returns an empty list if the character's node is not discovered.
            If the character's node is discovered, returns the list of starting index stored in the last node of the sequence.

        Input:
            sequence: The sequence to search for

        Return:
            A list of starting indexes of the sequence found

        Time complexity:
            Time complexity analysis: O(N), where N is the length of the sequence.
        Space complexity:
            Input space analysis: O(N) for the sequence.
            Aux space analysis: O(1) for traversal.
        """
        current = self.root  # Start at the root
        for char in sequence:  # Traverse the trie using the pattern characters
            index = ord(char) - ord('A') + 1  # Calculate the index for the character
            if current.link[index] is None:  # If no node exists for the character, pattern is not found
                return []
            current = current.link[index]  # Move to the next node
        return current.indexes  # Return starting index

    def find(self, start, end):
        """
        Function description:
            Finds all substrings in the genome that start with 'start' and end with 'end'.

        Approach description:
            Uses the suffix trie to locate the starting index of the start sequence.
            The function uses the find_substring function to find all starting index of the start and end sequence in the genome.
            It then iterates through the list of start and end found.
            For each starting index, searches for the end sequence starting from the node after the start sequence.
            If the end sequence exists, constructs a substring matching the start and end seqeuences then appends it to the output list.

        Input:
            start: The starting sequence
            end: The ending sequence

        Time complexity:
            Time complexity analysis: O(T + U + V),
            where T is the length of 'start', U is the length of 'end' and
            V is the number of characters in the output list.
        Space complexity:
            Input space analysis: O(T + U) for the start and end sequences.
            Aux space analysis: O(V) for the output list.

        Return:
            A list of substrings that start with 'start' and end with 'end'.
        """
        ret = []  # Output list containg list of substrings

        # Find starting index for the 'start' sequence
        start_indexes = self.find_sequence(start)
        # If the starting sequence don't exist in the genome
        # Just return an empty list as there's no point to go through the end sequence already
        if not start_indexes:
            return []

        # Find ending index for the end sequence
        end_indexes = self.find_sequence(end)

        for start_idx in start_indexes:  # Iterate over each start (T)
            # Iterate over each end (U)
            for end_idx in end_indexes:
                if end_idx >= start_idx + len(start):
                    substring = "" # Used for concatinating the substrings for the output
                    for i in range(start_idx, end_idx + len(end)): # Iterate over the length of each substring (V)
                        substring += self.genome[i]
                    ret.append(substring)

        return ret

# QUESTION 2: Securing the companies
# some notes : officers n, companies m, shift 3, days 30

class Vertex:
    """
    Class to represent a vertex in the flow graph.
    """

    def __init__(self, id):
        """
        Function description:
            Initializes the Vertex with a given ID.

        Input:
            id: The ID for the vertex.
        """
        self.id = id  # Vertex ID
        self.edges = []  # List to store edges connected to the vertex
        self.previous = None  # To track the path in BFS

    def add_edge(self, edge):
        """
        Function description:
            Adds an edge to the vertex's edge list.

        Input:
            edge: The edge to be added.
        """
        self.edges.append(edge)


class Edge:
    """
    Class to represent an edge in the flow network graph.
    """

    def __init__(self, u, v, w):
        """
        Function description:
            Initializes the Edge with source, destination, and capacity.

        Input:
            u: The source vertex of the edge.
            v: The destination vertex of the edge.
            w: The maximum capacity of the edge.
        """
        self.source_vertex = u  # Source vertex
        self.destination_vertex = v  # Destination vertex
        self.capacity = w  # Capacity of the edge
        self.flow = 0  # Initial flow is zero


class Graph:
    """
    Class to manage the flow network structure, adding edges, and computing maximum flow.
    """

    def __init__(self, num_officers, num_companies, preferences, officers_per_org, min_shifts, max_shifts):
        """
        Function Description:
            Initializes the flow graph with vertices, edges, and the required constraints.

        Approach Description:
            Builds a graph where vertices represent officers and companies. Edges are added
            based on officer preferences and company requirements, with source and sink vertices
            to facilitate the max flow calculation using the Edmonds-Karp algorithm.

        Input:
            num_officers: Number of security officers
            num_companies: Number of companies
            preferences: List of preferences for each security officer
            officers_per_org: List of required officers per company per shift
            min_shifts: Minimum number of shifts an officer can work
            max_shifts: Maximum number of shifts an officer can work

        Time complexity:
            Time complexity analysis: O(n + m), where n is the number of officers and m is the number of companies
            when creating the vertices.

        Space Complexity:
            Input Space analysis: O(n + m)
            Aux Space analysis: O(n * m)
        """
        self.edges = []
        self.num_officers = num_officers
        self.num_companies = num_companies
        self.preferences = preferences
        self.officers_per_org = officers_per_org
        self.min_shifts = min_shifts
        self.max_shifts = max_shifts
        self.vertices = []
        # Create vertces for officers, companies, source and sink
        for i in range(num_officers + num_companies + 2): # + 2 for source and sink vertex
            self.vertices.append(Vertex(i))
        self.source = num_officers + num_companies # source vertex
        self.sink = num_officers + num_companies + 1 # sink vertex
        self.build_graph() # Add the edges to the graph

    def build_graph(self):
        """
        Function Description:
            Builds the initial graph with the given officers, companies, and constraints.

        Approach Description:
            Constructs the graph by adding edges between source, officers, companies, and sink nodes
            based on preferences and requirements. The source connects to all officers, each company connects to the sink,
            and officers are connected to companies based on preferences.

        Time Complexity:
            Time Complexity analysis: O(3(n * m)), where n is the number of officers
            and m is the number of companies due to the nested for loops.

        Space Complexity:
            Input Space analysis: O(n + m)
            Aux Space analysis: O(3(n * m))
        """
        # Connect source to each officer with max_shifts capacity
        for i in range(self.num_officers):
            edge = Edge(self.source, i, self.max_shifts)  # Edge from source to officer
            self.vertices[self.source].add_edge(edge)
            self.edges.append(edge)

        # Connect each company to the sink for each shift
        for j in range(self.num_companies):
            for k in range(3):
                edge = Edge(self.num_officers + j, self.sink,
                            self.officers_per_org[j][k])  # Edge from company to sink for each shift
                self.vertices[self.num_officers + j].add_edge(edge)
                self.edges.append(edge)

        # Connect officers to companies based on preferences
        for i in range(self.num_officers):
            for j in range(self.num_companies):
                for k in range(3):
                    if self.preferences[i][k] == 1:
                        edge = Edge(i, self.num_officers + j, 1)  # Edge from officer to company based on preference
                        self.vertices[i].add_edge(edge)
                        self.edges.append(edge)

    def bfs(self):
        """
        Function description:
            Performs Breadth-First Search to find an augmenting path in the residual graph.
            This function utilises deque (double ended queue) from the collections library for quicker append and pop


        Approach description:
            Uses a queue to traverse the graph level by level, marking visited vertices and
            keeping track of the path using parent pointers.

        References used:
            [FIT2004_2020sem01] Lecture04 P1 Graph BFS DFS Lecture05 P1 Dijkstra

        Time complexity:
            Time complexity analysis: O(n + m), where n is the number of officers and m is the number of companies.

        Space complexity:
            Input space analysis: O(n)
            Aux space analysis: O(n)

        Return:
            A boolean statement: True if a path from source to sink is found, False otherwise.
        """
        # Initialize all vertices as undiscovered
        for vertex in self.vertices:
            vertex.discovered = False
            vertex.previous = None

        # Use deque for efficient popping from the front
        queue = deque([self.source])
        self.vertices[self.source].discovered = True

        while queue:
            u = queue.popleft()  # pop a vertex left side of the dequeue
            for edge in self.vertices[u].edges:  # Explore all edges of the dequeued vertex
                v = edge.destination_vertex
                # If the vertex is not discovered and has remaining capacity
                if not self.vertices[v].discovered and edge.capacity > edge.flow:
                    self.vertices[v].discovered = True
                    self.vertices[v].previous = edge  # Track the path
                    queue.append(v)
                    if v == self.sink:  # If we reach the sink
                        return True
        return False

    def max_flow(self):
        """
        References used:
            [FIT2004_2020sem01] Lecture08 FlowNetwork
            [FIT2004_2022sem01] Studio10 Q8 FlowNetwork DesignWithDemandLowerBound
            FIT2004 PASS Session Ford Fulkerson, Circulation with demand

        Function description:
            Computes the maximum flow in the flow graph using the Edmonds-Karp implementation
            of the Ford-Fulkerson algorithm.

        Approach description:
            Repeatedly finds augmenting paths using BFS and augments the flow along these paths
            until no more augmenting paths are found. The residual capacities of the edges are updated
            accordingly to reflect the augmented flow.

        Return:
            The maximum flow from source to sink.

        Time complexity:
            Time complexity analysis: O(VE^2), where V is the number of vertices and E is the number of edges.

        Space complexity:
            Input space analysis: O(V + E)
            Aux space analysis: O(V + E)
        """
        max_flow = 0

        # While there is an augmenting path, increment the flow
        while self.bfs():
            path_flow = float('Inf')
            s = self.sink

            # Find the maximum flow through the path found by BFS
            while s != self.source:
                edge = self.vertices[s].previous
                path_flow = min(path_flow, edge.capacity - edge.flow)
                s = edge.source_vertex if edge.destination_vertex == s else edge.destination_vertex

            # Update residual capacities of the edges and reverse edges along the path
            v = self.sink
            while v != self.source:
                edge = self.vertices[v].previous
                if edge.destination_vertex == v:
                    edge.flow += path_flow
                else:
                    edge.flow -= path_flow
                v = edge.source_vertex if edge.destination_vertex == v else edge.destination_vertex

            max_flow += path_flow

        return max_flow


def allocate(preferences, officers_per_org, min_shifts, max_shifts):
    """
    Function description:
        Allocates security officers to company shifts based on preferences and requirements.

    Approach description:
        Initializes a flow graph and computes the maximum flow using the Edmonds-Karp algorithm to determine
        the optimal allocation of officers to shifts while respecting the constraints.

    Input:
        preferences: List of preferences for each security officer.
        officers_per_org: List of required officers per company per shift.
        min_shifts: Minimum number of shifts an officer can work.
        max_shifts: Maximum number of shifts an officer can work.

    Time Complexity:
        Time complexity analysis: O(n * n * m),
        where n is the number of officers and m is the number of companies.

    Space Complexity:
        Input space analysis: O(n * m)
        Aux space analysis: O(n * m)

    Return:
        A 3D list representing the allocation of officers to companies and shifts.
    """
    num_officers = len(preferences)
    num_companies = len(officers_per_org)
    flow_graph = Graph(num_officers, num_companies, preferences, officers_per_org, min_shifts, max_shifts)

    max_possible_flow = flow_graph.max_flow()

    total_shifts_required = 0
    for row in officers_per_org:
        for shifts in row:
            total_shifts_required += shifts
    if max_possible_flow < total_shifts_required:
        return None

    # Initialize allocation structure
    allocation = []
    for i in range(num_officers):
        officer_alloc = []
        for j in range(num_companies):
            company_alloc = [0, 0, 0]  # Initialize allocation for each officer and company
            officer_alloc.append(company_alloc)
        allocation.append(officer_alloc)

    # Process each edge to create the final allocation
    for edge in flow_graph.edges:
        if edge.source_vertex < num_officers and edge.destination_vertex >= num_officers and edge.destination_vertex < num_officers + num_companies:
            officer = edge.source_vertex
            company = edge.destination_vertex - num_officers
            for shift in range(3):
                if preferences[officer][shift] == 1 and officers_per_org[company][shift] > 0:
                    allocation[officer][company][shift] = 1
                    officers_per_org[company][shift] -= 1
                    break

    return allocation



