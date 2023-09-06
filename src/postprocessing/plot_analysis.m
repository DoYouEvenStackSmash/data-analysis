file = '\\wsl.localhost\ubuntu\home\aroot\stuff\data-analysis\src\python-processing\search_tree_likelihoods.csv';
data_table = readtable(file);

file2 = '\\wsl.localhost\ubuntu\home\aroot\stuff\data-analysis\src\python-processing\all_pairs_likelihoods.csv';
data_table2 = readtable(file2);

error = abs(data_table.area_likelihood - data_table2.area_likelihood);

plots = tiledlayout(2, 1, 'TileSpacing','Compact');
ub = length(error) - 1
t = [0:ub];
k = data_table2.area_likelihood - mean(data_table2.area_likelihood);
title(plots, "likelihood data, C = 45 members");%, lambda val: 0.2727643646373728");
xlabel(plots, "image index")
ylabel(plots, "magnitude")
p2 = nexttile;
hold on;
grid on;
title(p2, ["sum over members likelihood"])
plot(p2, t, data_table.single_point_likelihood, 'LineWidth', 2);
plot(p2, t, data_table2.area_likelihood, 'LineWidth', 1.1);
xlim([0 ub])
legend(["cluster"], ["global scope"]);
hold off;
p3 = nexttile;
hold on
grid on
title(p3, ["error"]);
plot(p3, t, error, 'LineWidth', 2);
xlim([0 ub]);
legend(["error"]);
hold off;
% p1 = nexttile;
% hold on;
% grid on;
% title(p1, ["single term likelihoods"]);
% plot(p1, t, data_table.single_point_likelihood, 'LineWidth', 2);
% plot(p1, t, data_table2.single_point_likelihood, 'LineWidth', 1.1);
% legend(["tree search"], ["global search"]);
% xlim([0 ub]);
% hold off;

% p3 = nexttile;
% hold on;
% grid on;
% plot(p3, t, data_table.area_likelihood, 'LineWidth', 2);
% legend(["cluster likelihood"])
% xlim([0 ub])
% hold off;
% p4 = nexttile;
% hold on;
% grid on;
% plot(p4, t, data_table2.area_likelihood, 'LineWidth', 2);
% legend(["global likelihood"]);
% xlim([0 ub])
% hold off;
clear all;



