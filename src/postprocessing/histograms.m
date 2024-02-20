% Load data from CSV files
% path = '\\wsl.localhost\ubuntu\home\aroot\data\testify_likelihoods.csv'
path = '\\wsl.localhost\ubuntu\home\aroot\stuff\data-analysis\src\tree-likelihood\python\None_likelihoods.csv'
% files = []
% search_tree_likelihoods = readtable('\\wsl.localhost\ubuntu\home\aroot\stuff\data-analysis\src\python-processing\search_tree_likelihoods.csv');
% global_likelihoods = readtable('\\wsl.localhost\ubuntu\home\aroot\stuff\data-analysis\src\python-processing\all_pairs_likelihoods.csv');
likelihoods = readtable(path);

% Extract single point likelihood and area likelihood data
single_point_likelihoods = likelihoods.single_point_likelihood;% - mean(likelihoods.single_point_likelihood);
% single_point_likelihoods(isinf(single_point_likelihoods))=0;

area_likelihoods = likelihoods.area_likelihood;
area_likelihoods(isinf(area_likelihoods))=0;

v = mean(area_likelihoods); 
vs = mean(single_point_likelihoods);
% area_likelihoods = (area_likelihoods - v);
% single_point_likelihoods = single_point_likelihoods - vs;

spl_average = mean(single_point_likelihoods);
al_average = mean(area_likelihoods);
    
% Calculate the standard deviation of single point likelihoods and area likelihoods
spl_std_dev = std(single_point_likelihoods);
al_std_dev = std(area_likelihoods);

% Calculate bin widths using Scott's rule
spl_bin_width = 3.5 * spl_std_dev / (length(single_point_likelihoods)^(1/3));
al_bin_width = 3.5 * al_std_dev / (length(area_likelihoods)^(1/3));
cl_bin_width = 3.5 * std(maxmats) / (length(maxmats)^(1/3));
cl_bins = min(maxmats):cl_bin_width:max(maxmats);
cl_bin_count = ceil((abs(max(maxmats) - min(maxmats))) / cl_bin_width);

% Calculate the number of bins
spl_bin_count = (max(single_point_likelihoods) - min(single_point_likelihoods)) / spl_bin_width;
al_bin_count = ceil(abs((max(area_likelihoods) - min(area_likelihoods))) / al_bin_width);

% Create subplots and histograms
% plots = tiledlayout(4, 1);%, 'TileSpacing', 'Compact');
figure;
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
% p3 = nexttile;
subplot(2,2,[1,2])
plot( t, single_point_likelihoods,'b', 'LineWidth', 1);

hold on;
grid on;
plot( t, area_likelihoods, 'LineWidth', 1);
% plot( t, single_point_likelihoods,'b', 'LineWidth', 2);
title( 'Image Log-Likelihoods');
legend("Greedy likelihood", "Patient likelihood");
xlim([0 length(single_point_likelihoods) - 1]);
xlabel( 'Image Index');
ylabel( 'Log-Likelihood')
hold off;

% p1 = nexttile;
subplot(2,2,[3,4])
% bar( histcounts(single_point_likelihoods),'hist');
histogram(single_point_likelihoods, int64(spl_bin_count));
title(['Distribution of Log-Likelihoods']);%, ["Patient\tGreedy", ["Mean: "+ al_average, "std: " + al_std_dev] ['Greedy ', ["Mean: "+ spl_average, "std: " + spl_std_dev]]]);
disp([al_average, al_std_dev]);
disp([spl_average, spl_std_dev]);
display([mean(maxmats), std(maxmats)]);
% xlabel( 'Values');
% ylabel( 'Log-Like');
hold on;
% p2 = nexttile;
% subplot(3,2,[3,4])

histogram(area_likelihoods,int64(al_bin_count));
% hold off;
% [N,E] = histcounts(maxmats,cl_bin_count);

% h = histogram(data, edges, 'FaceColor', 'b', 'EdgeColor', 'none');


% counts = N;
% gap = (E(2) - E(1))/2
% centers = linspace(E(1)+gap, E(end)-gap, length(N));

% Extract histogram counts and bin centers
h = histogram(maxmats,cl_bin_count,"Visible","off");
counts = h.Values;
gap = (h.BinEdges(2) - h.BinEdges(1))/2;
centers = linspace(h.BinLimits(1)+gap,h.BinLimits(2)-gap,length(h.Values));
% centers = h.BinEdges(1:end-1) + diff(h.BinEdges)/2;
% centers = E(1:end-1) + diff(E)/2;

% Perform polynomial fit
degree = 9; % Choose the degree of the polynomial
p = polyfit(centers, counts, degree); % Fit polynomial
x_fit = linspace(min(centers), max(centers), 150); % Points for fitting line
y_fit = polyval(p, x_fit); % Evaluate polynomial at x_fit

% Plot polynomial fit line
% hold on;
y_fit(y_fit<0)=0;
plot(x_fit, y_fit, 'k-', 'LineWidth', 2);
% hold off;
ylabel("Frequency");
xlabel("Log-Likelihood")
legend("Greedy", "Patient","True");
grid on;
hold off;
% Create a table to display mean and std
% t = uitable('Data', {spl_average, spl_std_dev; al_average, al_std_dev}, ...
%             'RowName', {'Greedy', 'Patient'}, ...
%             'ColumnName', {'Mean', 'Standard Deviation'}, ...
%             'Units','normalized','Position', [0.55, 0.8, 0.3, 0.15]);
% t.FontSize=10;


% xlabel( 'Values');
% ylabel( 'Frequency');
% p4 = nexttile;
% 
% subplot(3,2,[5,6])
% % plot( t,error);
% % bar(histcounts(error), 'histc');
% title( 'Relative Error');
% xlabel( 'Values');
% ylabel( 'Magnitude');
% xlim([0, length(single_point_likelihoods) - 1]);

% % Add the additional plots
% ub = length(error) - 1;
% t = 0:ub;

% p4 = nexttile;
% hold on;
% grid on;
% title( 'Sum Over Members Likelihood');
% plot( t, search_tree_likelihoods.area_likelihood, 'LineWidth', 2);
% xlim( [0 length(single_point_likelihoods) - 1]);
% legend( {'Cluster Likelihood'});
% hold off;

% % Link the x-axis limits of the additional plots with the last subplot
% linkaxes([ p4], 'x');

% Set the overall title and axis labels
% title(plots, 'Log Likelihood Data, SNR=0.1');
% xlabel(plots, 'Image Index');
% ylabel(plots, 'Magnitude');

