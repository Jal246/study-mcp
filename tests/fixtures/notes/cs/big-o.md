# Big-O Notation

Describes how running time grows as input size n grows.

- O(1): constant, e.g. array index lookup, hash map get
- O(log n): binary search
- O(n): loop over a list once
- O(n log n): merge sort, heap sort
- O(n^2): nested loops, bubble sort

Rule: drop constants and smaller terms. O(3n^2 + 5n) is O(n^2).
