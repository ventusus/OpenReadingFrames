Open reading frames

In Molecular Genetics, there is a notion of an Open Reading Frame (ORF). An ORF is a
portion of DNA that is used as the blueprint for a protein. All ORFs start with a particular
sequence, and end with a particular sequence.

In this task, we wish to find all sections of a genome which start with a given sequence of
characters, and end with a (possibly) different given sequence of characters.

Input
genome is a single non-empty string consisting only of uppercase [A-D]. genome is passed as an
arguement to the __init__ method of OrfFinder (i.e. it gets used when creating an instance
of the class).
start and end are each a single non-empty string consisting of only uppercase [A-D].

start and end are each a single non-empty string consisting of only uppercase [A-D].

Output
find returns a list of strings. This list contains all the substrings of genome which have start
as a prefix and end as a suffix. There is no particular requirement for the order of these strings.
start and end must not overlap in the substring (see the last two cases of the example below).

Approach Description:

- Represents nodes in a suffix trie.
- Constructs a suffix trie from the genome string.
- Initializes the OrfFinder with the genome and constructs the suffix trie.
- Inserts a suffix into the trie starting from the given index.
- Iterates through the suffix, updates nodes, and stores starting indexes.
- Finds all starting indexes of a sequence in the trie.
- Traverses the trie to locate the sequence.
- Finds all substrings in the genome that start with start and end with end.
- Uses the suffix trie to find starting indexes of start and end.
- Constructs and returns substrings that match the criteria.

Complexity: 
Let T be the length of the string start, U be the length of the string end, and V be the
number of characters in the output list (for a correctly generated output list according to
the instructions in 1.2), then find must run in time complexity (T + U + V ).

As an example of what the complexity for find means, consider a string consisting of N/2 "B"s
followed by N/2 "A"s. If we call find("A","B"), the output is empty, so V is O(1). On the other
hand, if we call find("B", "A") then V is O(N^2).
