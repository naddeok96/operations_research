# Imports
import numpy as np
from prettytable import PrettyTable

# Functions
def optimality_test(tableau):
    """Performs optimality test

    Args:
        tableau (numpy.ndarray): Current tableau to perform optimality test on

    Returns:
        [bool]: If True then not optimal
    """
    return (tableau[0, :] < 0).any()

def get_basic_variables(tableau, tableau_names):
    """Determine the names of the basic variables

    Args:
        tableau (numpy.ndarray): Current tableau
        tableau_names (list of strings): Column names of tableau

    Returns:
        [list of strings]: Names of basic variables
    """
    # Determine size of basis
    basis_size = np.size(tableau, 0)

    # Check if column is a form of e
    basic_variables = []
    for i in range(basis_size):
        for j, column in enumerate(tableau.T):
            basis = np.zeros(basis_size)
            basis[i] = 1

            # If found stop
            if (column == basis).all():
                basic_variables.append(tableau_names[j])
                break

            # If missed try rounding
            if j == len(tableau.T) - 1:
                for k, column in enumerate(tableau.T):
                    column = np.round(column, decimals = 6)
                    basis = np.zeros(basis_size)
                    basis[i] = 1
                        
                    if (column == basis).all():
                        basic_variables.append(tableau_names[k])
                        break
                    if k == len(tableau.T) - 1:
                        print("missed the mark")
                        exit()

    return basic_variables

def get_BFS_and_Z(tableau, tableau_names, basic_variables):
    """Determine basic feasible solution and objective function output

    Args:
        tableau (numpy.ndarray): Current tableau
        tableau_names (list of strings): Column names of tableau
        basic_variables (list of strings): List of basic variables

    Returns:
        [tuple(list of strings, float)]: Tuple of basic feasible solution and objective function output
    """
    BFS = np.zeros(len(tableau_names) - 2)
    for i, basic_variable in enumerate(basic_variables):
        # Determine column of current basic_variable
        col_index = tableau_names.index(basic_variable)

        # Determine row that this basic variable represents
        row_index = np.where(np.round(tableau[:, col_index], decimals = 5) == 1)[0][0]

        # First basic variable is always Z
        if i == 0:
            Z = tableau[row_index, -1]
        else:
            BFS[col_index - 1] = tableau[row_index, -1]

    return BFS.tolist(), Z


def get_enter_exit(tableau, tableau_names, basic_variables):
    """Determine which variable is becoming a basic variable and which is leaving

    Args:
        tableau (numpy.array): Current tableau
        tableau_names (list of strings): Column names of tableau
        basic_variables (list of strings): Current basic variables

    Returns:
        [tuple(string, string)]: (entering variable, exiting variable)
    """

    # Find direction of biggest increase
    entering = tableau_names[1:-1][np.argmin(tableau[0, 1:-1])]

    # Minimum ratio test
    ratios = []
    for i, row in enumerate(tableau[:, 1:-1][1:, np.argmin(tableau[0, 1:-1])]):
        if row > 0:
            ratios.append(tableau[1:, -1][i] / row)
        else:
            ratios.append(1e5)
    exiting = basic_variables[np.argmin(ratios) + 1]

    return entering, exiting

def get_transition_matrix(tableau, tableau_names, basic_variables, entering, exiting):
    """Determines transition matrix based off entering and exiting variables

    Args:
        tableau (numpy.array): Current tableau
        tableau_names (list of strings): Column names of tableau
        basic_variables (list of strings): Current basic variables
        entering (string): Variable that is becoming a basic variable
        exiting (string): Variable that is exiting a basic variable

    Returns:
        [numpy.array]: Transition matrix
    """
    
    # Determine row and col of pivot element
    row_idx = basic_variables.index(exiting)
    col_idx = tableau_names.index(entering)
    pivot_element = tableau[row_idx, col_idx]

    # Calculate transition matrix
    transition_matrix = np.eye(np.size(tableau, 0))
    for i, row in enumerate(tableau[:, col_idx]):
        if i == row_idx:
            transition_matrix[i, row_idx] = 1 / pivot_element
        else:
            transition_matrix[i, row_idx] = -row / pivot_element

    return transition_matrix

def print_table(iteration, tableau, tableau_names, basic_variables):
    """Puts numpy array into a pretty table

    Args:
        iteration (int): The iteration number
        tableau (numpy.ndarray): Tableaus to be displayed
        column_names (list of strings): Column names of tableaus
        basic_variables (list of strings): List of basic variables
    """
    column_names = tableau_names.copy()
    column_names.insert(0, "")
    table = PrettyTable(column_names)
    for i, row in enumerate(tableau):
        table.add_row(np.append(basic_variables[i], row))
    
    print("tableau " + str(iteration + 1))
    print(table)



# Hyperparameters
tableau_names = ["Z", "x1", "x2", "x3", "x4", "x5", "x6", "RHS"]
max_iter = 10

# Matricies
tableaus = []
transitions = []

iteration = 0
p1 = np.array( [[1,     -15,     12, -6, 8,      0, 0,      0],
                           [0,    5/17,  12/17,  0, 1,  -1/17, 0,  84/17],
                           [0,  -13/34,  13/34,  1, 0,   3/17, 0,   3/17],
                           [0,  201/34, -65/34,  0, 0, -15/17, 1, 240/17]])
reestablish = np.array([[1, -8, 6, 0],
                        [0,  1, 0, 0],
                        [0,  0, 1, 0],
                        [0,  0, 0, 1]])        

tableaus.append(np.dot(reestablish, p1))

while optimality_test(tableaus[iteration]):
    # Determine basic variables
    basic_variables = get_basic_variables(tableaus[iteration], tableau_names)

    # Display current tableau
    print_table(iteration, tableaus[iteration], tableau_names, basic_variables)

    # Determine BFS and Z
    BFS, Z = get_BFS_and_Z(tableaus[iteration], tableau_names, basic_variables)
    print("BFS: ", BFS)
    print("Z: ", Z)

    # Determine exiting and entering variables
    entering, exiting = get_enter_exit(tableaus[iteration], tableau_names, basic_variables)
    print("Entering: ", entering)
    print("Exiting: ", exiting)

    # Determine transition matrix
    transitions.append(get_transition_matrix(tableaus[iteration], tableau_names, basic_variables, entering, exiting))

    # Update tableaus
    new_tableau = np.dot(transitions[iteration], tableaus[iteration])
    new_tableau[abs(new_tableau) < 1e-10] = 0
    tableaus.append(new_tableau)

    # Increase iteration
    iteration += 1

    # Safety
    if iteration == max_iter:
        exit()
# Determine basic variables
basic_variables = get_basic_variables(tableaus[iteration], tableau_names)

# Display current tableau
print_table(iteration, tableaus[iteration], tableau_names, basic_variables)

# Determine BFS and Z
BFS, Z = get_BFS_and_Z(tableaus[iteration], tableau_names, basic_variables)
print("BFS: ", BFS)
print("Z: ", Z)

                