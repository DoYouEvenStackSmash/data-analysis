% Load data from CSV files
% path = '\\wsl.localhost\ubuntu\home\aroot\data\testify_likelihoods.csv'
path = '\\wsl.localhost\ubuntu\home\aroot\stuff\data-analysis\src\tree-likelihood\python\None_likelihoods.csv'
% files = []
% search_tree_likelihoods = readtable('\\wsl.localhost\ubuntu\home\aroot\stuff\data-analysis\src\python-processing\search_tree_likelihoods.csv');
% global_likelihoods = readtable('\\wsl.localhost\ubuntu\home\aroot\stuff\data-analysis\src\python-processing\all_pairs_likelihoods.csv');
likelihoods = readtable(path);

% Extract single point likelihood and area likelihood data
single_point_likelihoods = likelihoods.single_point_likelihood;% - mean(likelihoods.single_point_likelihood);
single_point_likelihoods(isinf(single_point_likelihoods))=0;
area_likelihoods = likelihoods.area_likelihood ;
area_likelihoods(isinf(area_likelihoods))=0;

spl_average = mean(single_point_likelihoods);
al_average = mean(area_likelihoods);
    
% Calculate the standard deviation of single point likelihoods and area likelihoods
spl_std_dev = std(single_point_likelihoods);
al_std_dev = std(area_likelihoods);

% Calculate bin widths using Scott's rule
spl_bin_width = 3.5 * spl_std_dev / (length(single_point_likelihoods)^(1/3));
al_bin_width = 3.5 * al_std_dev / (length(area_likelihoods)^(1/3));

% Calculate the number of bins
spl_bin_count = (max(single_point_likelihoods) - min(single_point_likelihoods)) / spl_bin_width;
al_bin_count = (max(area_likelihoods) - min(area_likelihoods)) / al_bin_width;

% Create subplots and histograms
plots = tiledlayout(4, 1);%, 'TileSpacing', 'Compact');

% Plot single point likelihood histogram
spl_bins = min(single_point_likelihoods):spl_bin_width:max(single_point_likelihoods);
spl_hist = histc(single_point_likelihoods, spl_bins);

% Plot area likelihood histogram
al_bins = min(area_likelihoods):al_bin_width:max(area_likelihoods);
al_hist = histc(area_likelihoods, al_bins);

t = [0:length(single_point_likelihoods) - 1];

% Plot error
error = abs(single_point_likelihoods - area_likelihoods) ./ abs(area_likelihoods);
error_average = mean(error)
error_std_dev = std(error);
error_bin_width = 3.5 * error_std_dev / (length(error)^(1/3));
error_bins = min(error):error_bin_width:max(error);
error_hist = histc(error, error_bins);

% Plot subplots
p3 = nexttile;
hold on;
grid on;
plot(p3, t, single_point_likelihoods,'b', 'LineWidth', 1);
plot(p3, t, area_likelihoods, 'LineWidth', 0.2);
% plot(p3, t, single_point_likelihoods,'b', 'LineWidth', 2);
title(p3, 'Image Log-Likelihoods');
legend("Greedy likelihood", "Bounded likelihood");
xlim([0 length(single_point_likelihoods) - 1]);
xlabel(p3, 'Image Index');
ylabel(p3, 'Log-Likelihood')
hold off;

p1 = nexttile;
bar(p1, spl_bins, spl_hist, 'histc');
title(p1, 'Tree Approximated Log-Likelihood', ["Mean: "+ spl_average, "std: " + spl_std_dev]);
xlabel(p1, 'Values');
ylabel(p1, 'Frequency');

p2 = nexttile;
bar(p2, al_bins, al_hist, 'histc');
title(p2, 'True Log-Likelihood', ["Mean: "+ al_average, "std: " + al_std_dev]);
xlabel(p2, 'Values');
ylabel(p2, 'Frequency');
p4 = nexttile;
plot(p4, t,error);
% bar(p4, error_bins, error_hist, 'histc');
title(p4, 'Relative Error');
xlabel(p4, 'Values');
ylabel(p4, 'Magnitude');
xlim([0, length(single_point_likelihoods) - 1]);

% % Add the additional plots
% ub = length(error) - 1;
% t = 0:ub;

% p4 = nexttile;
% hold on;
% grid on;
% title(p4, 'Sum Over Members Likelihood');
% plot(p4, t, search_tree_likelihoods.area_likelihood, 'LineWidth', 2);
% xlim(p4, [0 length(single_point_likelihoods) - 1]);
% legend(p4, {'Cluster Likelihood'});
% hold off;

% % Link the x-axis limits of the additional plots with the last subplot
% linkaxes([p3, p4], 'x');

% Set the overall title and axis labels
title(plots, 'Log Likelihood Data, SNR=0.1');
% xlabel(plots, 'Image Index');
% ylabel(plots, 'Magnitude');

