import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
colors = list(set(mcolors.CSS4_COLORS))
# Function to calculate polar angle between points
def polar_angle(p0, p1):
    return np.arctan2(p1[1] - p0[1], p1[0] - p0[0])

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

def display_convex_hulls(clusters):
    height = len(clusters)
    colors.reverse()
    ci = 0
    plt.figure(facecolor='black')
    plt.gca().set_facecolor('black')
    for i,points in enumerate(reversed(clusters)):
        width = len(points)
        # print(width)
        # List of numpy arrays representing points
        # points = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6]), np.array([7, 8])]

        # Find the convex hull
        convex_hull_points = graham_scan_convex_hull(points) if len(points) > 2 else list(points)
        convex_hull_points.append(convex_hull_points[0])
        # Plot the original points
        # plt.scatter(*zip(*points), color='blue', label='Points')

    # Plot the convex hull
        x_coords = [p[0] for p in convex_hull_points]
        y_coords = [p[1] for p in convex_hull_points]
        # hull_points = [arr[0],arr[1] convex_hull_points# Closing the loop
        # print(hull_points)
        plt.plot(x_coords, y_coords, color=colors[(i) % len(colors)],linewidth=len(x_coords)/10)

    plt.xlabel('X')
    plt.ylabel('Y')
    # plt.background()
    plt.title('Convex Hull using Graham Scan')
    # plt.legend()
    plt.show()

def draw_cluster_points(pointlist):
    pl = []
    for points in pointlist:
        x_coords = [point[0] for point in points]
        y_coords = [point[1] for point in points]

        pl.append([x_coords, y_coords])
    # Generate a list of unique colors for each point
    colors = plt.cm.rainbow(np.linspace(0, 1, len(pl)))

    # Plot each point with a different color
    for p in range(len(pl)):
        x_coords = pl[p][0]
        y_coords = pl[p][1]
        i = p
        for j in range(len(x_coords)):
            plt.scatter(
                x_coords[j], y_coords[j], color=colors[i]
            )  # , label=f'Point {i+1}')

    # Add labels and legend
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Scatter Plot of Points")
    # plt.legend()

    # Show the plot
    plt.show()
