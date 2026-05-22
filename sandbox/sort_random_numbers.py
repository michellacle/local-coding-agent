import random

# Generate a list of 10 random numbers
random_numbers = [random.randint(0, 100) for _ in range(10)]

print("Original list:")
print(random_numbers)

# Sort the list
sorted_numbers = sorted(random_numbers)

print("\nSorted list (ascending):")
print(sorted_numbers)