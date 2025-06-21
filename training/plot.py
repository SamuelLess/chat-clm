import matplotlib.pyplot as plt

# Read numbers from stats.txt
numbers = []
with open('../stats.txt', 'r') as file:
    for line in file:
        try:
            numbers.append(float(line.strip()))
        except ValueError:
            pass  # Skip lines that cannot be converted to a number


# average 
average = sum(numbers) / len(numbers) if numbers else 0
print(f"Average: {average}")
# Plot the histogram
plt.hist(numbers, bins=1000, edgecolor='black')
plt.title('Histogram of Numbers')
plt.xlabel('Value')
plt.ylabel('Frequency')
# xlimit = 0.02
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()