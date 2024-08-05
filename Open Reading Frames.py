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
