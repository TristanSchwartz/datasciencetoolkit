from typing import List, Tuple

################################################################################
# Basic Recursive Knapsack with Item Tracking (Inefficient but Least Memory)
# Not Recommended
# Only for educational purposes
################################################################################

def knapsack_recursive(capacity: int, weights: List[int], values: List[int], n: int) -> Tuple[int, List[int]]:
    """
    Solves the knapsack problem using basic recursion and tracks selected items.
    This is the most basic version of the knapsack algorithm and is inefficient for large inputs.
    It is included here for educational purposes to show how recursion can be used to solve the problem.

    N: Number of items

    Time Complexity: O(2^n) - Exponential time due to branching for each item.
    Space Complexity: O(n) - Only call stack is used for recursion depth.


    Args:
        capacity (int): The maximum weight the knapsack can carry.
        weights (List[int]): The list of item weights.
        values (List[int]): The list of item values.
        n (int): Number of items being considered.

    Returns:
        Tuple[int, List[int]]: Maximum value and list of selected item indices.
    """
    if n == 0 or capacity == 0:
        return 0, []

    if weights[n - 1] > capacity:
        return knapsack_recursive(capacity, weights, values, n - 1)

    # Option 1: Exclude item
    exclude_value, exclude_items = knapsack_recursive(capacity, weights, values, n - 1)

    # Option 2: Include item
    include_value, include_items = knapsack_recursive(capacity - weights[n - 1], weights, values, n - 1)
    include_value += values[n - 1]

    # Choose the better option
    if include_value > exclude_value:
        return include_value, include_items + [n - 1]
    else:
        return exclude_value, exclude_items



################################################################################
# Standard DP 2D Knapsack with Item Tracking (Efficient but Most Memory)
# Not Recommended
# Only for educational purposes
################################################################################

def knapsack_2d(capacity: int, weights: List[int], values: List[int]) -> Tuple[int, List[int]]:
    """
    Solves the knapsack problem using DP (2D table) and tracks selected items.
    This is the standard dynamic programming approach to the knapsack problem.
    It uses a 2D array to store the maximum value for each weight and item combination.
    This is more efficient than the recursive approach but uses more memory.

    N: Number of items
    W: Maximum weight of the knapsack
    Time Complexity: O(nW) - Iterates through all items and weight capacities.
    Space Complexity: O(nW) - Stores values for every combination of item and weight.
    


    Args:
        capacity (int): The maximum weight the knapsack can carry.
        weights (List[int]): The list of item weights.
        values (List[int]): The list of item values.

    Returns:
        Tuple[int, List[int]]: Maximum value and list of selected item indices.
    """
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    # Build DP table
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], values[i - 1] + dp[i - 1][w - weights[i - 1]])
            else:
                dp[i][w] = dp[i - 1][w]

    # Backtrack to find selected items
    selected_items = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:  # If value changed, item was included
            selected_items.append(i - 1)
            w -= weights[i - 1]

    selected_items.reverse()
    return dp[n][capacity], selected_items


################################################################################
# Space-Optimized 1D DP Knapsack with Item Tracking (Efficient & Less Memory)
# Bounded and Unbounded Knapsack
# Recommended for Practical Use
# This is the most efficient version of the knapsack algorithm.
################################################################################

def knapsack_1d(items: List[Tuple[str, int, int, int]], capacity: int, knapsack_type: str = "bounded") -> Tuple[int, List[str]]:
    """
    Solves the knapsack problem using a space-optimized dynamic programming approach (1D array)
    while tracking selected items. It supports both bounded and unbounded knapsack problem variants.

    - For **bounded knapsack**, each item can be taken multiple times but with a specific quantity limit.
    - For **unbounded knapsack**, each item can be taken any number of times.

    Time Complexity (Bounded): O(nW * Q) - Where n is the number of items, W is the capacity, and Q is the maximum quantity of any item.
    Time Complexity (Unbounded): O(nW) - Each item can be taken unlimited times.

    Space Complexity: O(W) - Uses only a 1D array for maximum values at each weight.

    Args:
        items (List[Tuple[str, int, int, int]]): A list of items, where each item is represented as a tuple
                                                 (name, weight, value, quantity).
        capacity (int): The maximum weight the knapsack can carry.
        knapsack_type (str): The type of knapsack problem ('bounded' or 'unbounded').

    Returns:
        Tuple[int, List[str]]: The maximum value and a list of selected item names.
    """
    n = len(items)
    dp = [0] * (capacity + 1)  # DP array to store maximum values for each weight
    keep = [[] for _ in range(capacity + 1)]  # To track the items selected at each weight

    # Bounded knapsack: Iterate over each item and each possible quantity
    if knapsack_type == "bounded":
        for i in range(n):
            weight, value, quantity = items[i][1], items[i][2], items[i][3]
            for qty in range(1, quantity + 1):  # For each possible quantity of item i
                for w in range(capacity, weight * qty - 1, -1):  # Traverse capacity from back to avoid overwriting
                    if dp[w - weight * qty] + value * qty > dp[w]:
                        dp[w] = dp[w - weight * qty] + value * qty
                        keep[w] = keep[w - weight * qty][:] + [items[i][0]] * qty

    # Unbounded knapsack: For each item, you can take it an unlimited number of times
    elif knapsack_type == "unbounded":
        for i in range(n):
            weight, value, _ = items[i][1], items[i][2], items[i][3]
            for w in range(weight, capacity + 1):  # Traverse capacity from front to avoid overwriting
                if dp[w - weight] + value > dp[w]:
                    dp[w] = dp[w - weight] + value
                    keep[w] = keep[w - weight][:] + [items[i][0]]

    else:
        raise ValueError("Invalid knapsack type. Please choose either 'bounded' or 'unbounded'.")

    # Return the maximum value and the list of selected items
    return dp[capacity], keep[capacity]

#################################################################################
# Calling the function to test them
#################################################################################

# List of items (name, weight, value, quantity)
items = [
    ("Laptop", 3, 2000, 2),  
    ("Headphones", 1, 300, 5),  
    ("Water Bottle", 2, 150, 3),  
    ("Notebook", 1, 100, 10),  
    ("Smartphone", 1, 900, 1),  
    ("Jacket", 4, 500, 2),  
    ("Camera", 3, 1200, 1),  
    ("Shoes", 2, 700, 3),  
    ("Tablet", 2, 850, 4),  
    ("Sunglasses", 1, 250, 6)
]

# Adjustable backpack capacity
backpack_capacity = 7

# Choose 'bounded' or 'unbounded' for the knapsack problem type
knapsack_type = 'bounded'  # or 'unbounded'

# Example Usage
maximized_value, selected_items = knapsack_1d(items, backpack_capacity, knapsack_type)
print("Maximized Value:", maximized_value)
print("Selected Items:", selected_items)
