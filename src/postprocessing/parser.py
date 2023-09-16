#!/usr/bin/python3
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
colors = list(set(mcolors.CSS4_COLORS))

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

def display_clusters(clusters, centroids):
    """
    Displays the clusters with the corresponding centroids and enclosing convex hull using matplotlib
    """
    ci = 0
    reshaped_centers = centroids.reshape(-1, 2)
    cx_coords = reshaped_centers[:, 0]
    cy_coords = reshaped_centers[:, 1]
    # fig = plt.figure()
    # fig.set_facecolor("black")
    for i, m in enumerate(clusters):
        # reshaped_data = m.reshape(-1, 2)
        reshaped_data = m
        print(m.shape)
        if i + ci >= len(colors):
            ci = -len(colors)
        # Compute the convex hull using Graham Scan
        if len(m) > 2:
            convex_hull = graham_scan_convex_hull(reshaped_data)
            convex_hull.append(convex_hull[0])

            # Convert the convex hull points to a numpy array for visualization
            convex_hull_array = np.array(convex_hull)

            # Plot the original points and the convex hull
            plt.plot(
                convex_hull_array[:, 0],
                convex_hull_array[:, 1],
                c=colors[i + ci],
                marker="x",
                linewidth=len(m)
                / 40,  # adjust width of the line according to the hierarchy
                label="Convex Hull",
            )
        plt.scatter(
            reshaped_data[:, 0],
            reshaped_data[:, 1],
            c=colors[i + ci],
            label="Points",
            s=1,
        )
        x_coords = reshaped_data[:, 0]
        y_coords = reshaped_data[:, 1]
        plt.scatter(x_coords, y_coords, marker="o", color=colors[i + ci], s=1)
        plt.scatter(cx_coords[i], cy_coords[i], marker="x", color=colors[i + ci], s=10)
    plt.show()

def main():
  f = open("output.json", "r")
  s = json.loads(f.read())
  f.close()
  centroids = []
  clusters = []
  for i,c in enumerate(s["clusters"]):
    cluster = c[f"{i}"]
    centroid = np.array([cluster[0]["x"],cluster[0]["y"]])
    centroids.append(centroid)
    clst = []
    for pt in cluster[1:]:
      clst.append(np.array([pt["x"],pt["y"]]))
    clusters.append(np.array(clst))
  centroids = np.array(centroids)
  # print(clusters)
  display_clusters(clusters, centroids)
    # centroids.append(np.array([c[0]["x"],c[0]["y"]]))

    # cluster = np.array([c[i]["x"],c[i]["y"]])

# print(json.dumps(s, indent=2))
if __name__ == '__main__':
  main()