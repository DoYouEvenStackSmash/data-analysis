# from clustering_imports import *
import matplotlib.pyplot as plt
from geometry_fxns import *
from clustering_imports import *

def display_dendrogram(clusters, centroids):
    """
    Displays the clusters with the corresponding centroids and enclosing convex hull using matplotlib
    """
    reshaped_centers = centroids.reshape(-1, 2)
    cx_coords = reshaped_centers[:, 0]
    cy_coords = reshaped_centers[:, 1]

    # Prepare data for linkage and dendrogram
    all_data = np.concatenate(clusters)
    Z = linkage(all_data, "ward")

    plt.figure(figsize=(10, 5))

    # Plot dendrogram
    dendrogram(Z)

    # Plot data points and centroids
    for i, m in enumerate(clusters):
        reshaped_data = m.reshape(-1, 2)
        plt.scatter(
            reshaped_data[:, 0],
            reshaped_data[:, 1],
            c=colors[i],
            label="Cluster " + str(i + 1),
            s=10,
        )
        plt.scatter(cx_coords[i], cy_coords[i], marker="x", color=colors[i], s=50)

    plt.xlabel("Data Points")
    plt.ylabel("Distance")
    plt.title("Dendrogram of Hierarchical Clustering")
    plt.legend()
    plt.show()


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
                / 10,  # adjust width of the line according to the hierarchy
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


def generate_dot_graph(xy_pairs):
    dot_graph = "digraph G {\n"

    # Add nodes
    nodes = set()
    for x, y in xy_pairs:
        nodes.add(x)
        nodes.add(y)

    for node in nodes:
        dot_graph += f"    {node};\n"

    # Add edges
    for x, y in xy_pairs:
        dot_graph += f"    {x} -> {y};\n"

    dot_graph += "}"
    return dot_graph


def display_correlation_matrix(A, B, coefficients):
    n = len(A)
    corr_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            corr_matrix[i, j] = coefficients[i][j]
    
    fig, ax = plt.subplots()
    cax = ax.matshow(corr_matrix, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.xticks(range(n), A, rotation=90)
    plt.yticks(range(n), B)
    plt.colorbar(cax)

    plt.show()