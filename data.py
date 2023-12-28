import csv
import random


# what actually needs to happen for a wyckoff pattern:

# - initial maxima and minima (trading range)
# - minimas close together in series
# - minima that sweeps the previous minimas (goes lower)
# - maxima above initial maxima


# each random is a random amount of randoms **


# initial low
# random - between high and low
# initial high
# random between st and high
# ST
# random between st and high
# ST
# random between st and high
# random between sweep and st
# sweep (below initial low)
# random between sweep and st
# price above original range 
# random between higher price and initial high 


def append_random_numbers(my_list, num_to_append, lower_limit, upper_limit):
    """
    Append a given number of random numbers within a specified range to a list.

    Parameters:
    - my_list (list): The list to which random numbers will be appended.
    - num_to_append (int): The number of random numbers to append.
    - lower_limit (float): The lower limit of the random number range.
    - upper_limit (float): The upper limit of the random number range.
    """
    for _ in range(num_to_append):
        random_number = random.uniform(lower_limit, upper_limit)
        my_list.append(random_number)

# Example usage:
my_numbers = []
append_random_numbers(my_numbers, 5, 1, 10)

print(my_numbers)
