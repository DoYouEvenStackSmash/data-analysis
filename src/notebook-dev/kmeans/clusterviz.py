import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def graham_scan_convex_hull(points):
    def cross_product(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (r[0] - p[0]) * (q[1] - p[1])

    def polar_angle(p, q):
        return np.arctan2(q[1] - p[1], q[0] - p[0])

    # Find the point with the lowest y-coordinate (and leftmost if tied)
    start_point = min(points, key=lambda p: (p[1], p[0]))

    # Sort the points based on polar angles with respect to the start point
    sorted_points = sorted(
        points, key=lambda p: (polar_angle(start_point, p), -p[1], p[0])
    )

    # Initialize the convex hull with the first three sorted points
    convex_hull = [sorted_points[0], sorted_points[1], sorted_points[2]]

    # Iterate through the sorted points and construct the convex hull
    for point in sorted_points[3:]:
        while (
            len(convex_hull) >= 2
            and cross_product(convex_hull[-2], convex_hull[-1], point) <= 0
        ):
            convex_hull.pop()
        convex_hull.append(point)

    return convex_hull


# Reshape the array to (100, 2) for plotting
colors = list(set(mcolors.CSS4_COLORS))[::4]
# print(colors)
reshaped_centers = centers.reshape(-1, 2)
cx_coords = reshaped_centers[:, 0]
cy_coords = reshaped_centers[:, 1]

for i, M in enumerate(reclus):
    reshaped_data = M.reshape(-1, 2)
    print(reshaped_data.shape)
    # Compute the convex hull using Graham Scan
    convex_hull = graham_scan_convex_hull(reshaped_data)
    convex_hull.append(convex_hull[0])

    # Convert the convex hull points to a numpy array for visualization
    convex_hull_array = np.array(convex_hull)

    # Plot the original points and the convex hull
    # plt.scatter(reshaped_data[:, 0], reshaped_data[:, 1], c='blue', label='Points')
    plt.plot(
        convex_hull_array[:, 0],
        convex_hull_array[:, 1],
        c=colors[i],
        marker="o",
        label="Convex Hull",
    )

    x_coords = reshaped_data[:, 0]
    y_coords = reshaped_data[:, 1]
    plt.scatter(x_coords, y_coords, marker="o", color=colors[i], s=2)
    plt.scatter(cx_coords[i], cy_coords[i], marker="x", color=colors[i], s=30)

# print(clusters.shape)
# Extract x and y coordinates


# Create a scatter plot


# Add labels and title
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Scatter Plot of Data Points")

# Display the plot
plt.show()
