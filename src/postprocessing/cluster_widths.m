% Read the CSV file into a table
data = readtable('\\wsl.localhost\ubuntu\home\aroot\stuff\data-analysis\src\tree-likelihood\python\stats.csv');

% Extract the values from the two columns
col1 = data.level+1;
col2 = data.width;

% Fit a polynomial of degree N to the data
degree = 9; % Adjust the degree as needed
coefficients = polyfit(col1, col2, degree);

% Create a polynomial function using the coefficients
poly_function = polyval(coefficients, col1);

% Plot the original data using a violin plot
figure;
boxplot(col2, col1, 'Colors', 'b', 'Widths', 0.7);
hold on;

% Plot the polynomial fit
semilogy(col1, poly_function, 'r', 'LineWidth', 1.5);
grouped_means = splitapply(@mean, col2, col1);
for i = 1:length(grouped_means)
    text(i, grouped_means(i), sprintf('%.5f', grouped_means(i)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
end

% Plot the polynomial fit
scatter(col1, poly_function, 50, 'r', 'filled');
title('Cluster Widths with Polynomial Fit');
xlabel('Tree Level');
ylabel('Cluster Width');
legend('Polyfit', 'Mean Values');
grid on;
hold off;
