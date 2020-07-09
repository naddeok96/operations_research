# Imports
import numpy as np
from prettytable import PrettyTable

# Declare Variables
def print_table(M, column_names):
    """Puts numpy array into a pretty table

    Args:
        M (numpy.ndarray): Matrix to be displayed
        column_names (list of strings): Column names of matrix
    """
    table = PrettyTable(column_names)
    for row in M:
        table.add_row(row)
    print(table)

# Hyperparameters
tableau_names = ["Z", "x1", "x2", "x3", "x4", "x5", "x6", "RHS"]
transition_names = ["Z", "BV1", "BV2", "BV3", "BV4"]

# Tableau
M1 = np.array( [[1, -15, -14, 0, 0, 0, 0,   0],
                [0,  20,  10, 1, 0, 0, 0, 150],
                [0,  12,   8, 0, 1, 0, 0,  96],
                [0,   3,   4, 0, 0, 1, 0,  40],
                [0,   0,   1, 0, 0, 0, 1,   9]])
print("M1")
print_table(M1, tableau_names)
                
t1 = np.array( [[1,  15/20, 0, 0, 0],
                [0,   1/20, 0, 0, 0],
                [0, -12/20, 1, 0, 0],
                [0, - 3/20, 0, 1, 0],
                [0,      0, 0, 0, 1]])
print("t1")
print_table(t1, transition_names)

M2 = np.dot(t1, M1)
print("M2")
print_table(M2, tableau_names)

t2 = np.array( [[1, 0, 13/4, 0, 0],
                [0, 1, -1/4, 0, 0],
                [0, 0,  1/2, 0, 0],
                [0, 0, -5/4, 1, 0],
                [0, 0, -1/2, 0, 1]])
print("t2")
print_table(t2, transition_names)

M3 = np.dot(t2, M2)
print("M3")
print_table(M3, tableau_names)

t3 = np.array( [[1, 0, 0,    2, 0],
                [0, 1, 0, -1/3, 0],
                [0, 0, 1,  1/2, 0],
                [0, 0, 0,  5/3, 0],
                [0, 0, 0, -1/2, 1]])
print("t3")
print_table(t3, transition_names)

M4 = np.dot(t3, M3)
print("M4")
print_table(M4, tableau_names)

