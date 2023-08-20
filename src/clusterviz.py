import matplotlib.pyplot as plt

# Reshape the array to (100, 2) for plotting
reshaped_data = M.reshape(-1, 2)
reshaped_centers = centers.reshape(-1, 2)

# Extract x and y coordinates
x_coords = reshaped_data[:, 0]
y_coords = reshaped_data[:, 1]
cx_coords = reshaped_centers[:, 0]
cy_coords = reshaped_centers[:, 1]


# Create a scatter plot
plt.scatter(x_coords, y_coords, marker="o", color="b", s=10)
plt.scatter(cx_coords, cy_coords, marker="o", color="g", s=20)

# Add labels and title
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Scatter Plot of Data Points")

# Display the plot
plt.show()
