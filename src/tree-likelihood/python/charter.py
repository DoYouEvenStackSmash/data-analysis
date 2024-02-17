import plotly.graph_objects as go

# Sample data (parent-child pairs)
data = [
    ("A", "B"),
    ("A", "C"),
    ("B", "D"),
    ("B", "E"),
    ("C", "F"),
    ("C", "G"),
    ("D", "H"),
    ("D", "I"),
    ("E", "J"),
]

# Create a dictionary to store the hierarchy
hierarchy = {}

# Build the hierarchy
for parent, child in data:
    print(hierarchy)
    if parent not in hierarchy:
        hierarchy[parent] = {"name": parent, "children": []}
    if child not in hierarchy:
        hierarchy[child] = {"name": child}
    hierarchy[parent]["children"].append(hierarchy[child])

# Create the sunburst figure
fig = go.Figure(
    go.Sunburst(
        ids=[key for key in hierarchy],
        labels=[hierarchy[key]["name"] for key in hierarchy],
        parents=["" if key == "root" else key for key in hierarchy],
        values=[2] * len(hierarchy),  # All nodes have the same value for simplicity
    )
)

# Update layout
fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))

# Show the chart
fig.show()
