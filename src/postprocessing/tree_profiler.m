% Load the CSV data
data = readtable('\\wsl.localhost\ubuntu\home\aroot\stuff\data-analysis\src\python-processing\three_tree.csv'); % Replace 'your_data.csv' with the actual file path

% Extract the columns of interest
N = data.N;
D = data.D;
k = data.k;
R = data.R;
C = data.C;
tree_build = data.tree_build;

% Calculate the product N * D * k * R * C
% product = N .* D .* k .* R .* C;
% time = [0::8]
% Create a scatter plot of the product against tree_build
hold on;
grid on;
plot(N, tree_build, "-o", 'LineWidth', 2);
% Add labels and a title to the plot
xlabel('Number of structures');
ylabel('Time (seconds)');
title("Hierarchical clustering runtime against structure count");
hold off;
% title('Scatter Plot of N * D * k * R * C against tree\_build');
