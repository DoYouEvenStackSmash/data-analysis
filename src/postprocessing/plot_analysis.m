file = '\\wsl.localhost\ubuntu\home\aroot\stuff\data-analysis\src\python-processing\search_tree_likelihoods.csv';
data_table = readtable(file);

file2 = '\\wsl.localhost\ubuntu\home\aroot\stuff\data-analysis\src\python-processing\all_pairs_likelihoods.csv';
data_table2 = readtable(file2);

error = abs(data_table.single_point_likelihood - data_table2.single_point_likelihood);

plots = tiledlayout(4, 1, 'TileSpacing','Compact');
t = [0:255];

title(plots, "likelihood data");
p1 = nexttile;
hold on;
grid on;
plot(p1, t, data_table.single_point_likelihood, 'LineWidth', 2);
plot(p1, t, data_table2.single_point_likelihood);
legend(["tree search"], ["global search"]);
xlim([0 255]);
hold off;
p2 = nexttile;
hold on;
grid on;
plot(t, error, 'LineWidth', 2);
xlim([0 255])
legend(["error"]);
hold off;
p3 = nexttile;
hold on;
grid on;
plot(p3, t, data_table.area_likelihood, 'LineWidth', 2);
legend(["cluster likelihood"])
xlim([0 255])
hold off;
p4 = nexttile;
hold on;
grid on;
plot(p4, t, data_table2.area_likelihood, 'LineWidth', 2);
legend(["global likelihood"]);
xlim([0 255])
hold off;
clear all;



