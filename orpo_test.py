import math

# Set the number of decimal places you want to calculate
decimal_places = 15

# Calculate the value of pi using the Taylor series
pi = 0
for i in range(decimal_places + 1):
    pi += (4 / (8 * i - 6))

print("Pi to", decimal_places, "decimal places is:", round(pi, decimal_places))