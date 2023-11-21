% Given data
images = [1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680];
times = [0.912749770010123, 2.989286013005767, 2.653218198000104, 3.677179874997819, 4.971076945003006, 4.609119443004602, 4.43832607899094, 5.015284392007743, 6.4422190339973895, 7.567061241003103, 7.886000758007867, 8.607719869993161, 9.072583952001878, 8.452264118997846];
other_times = [13.085189700999763
22.20253994000086
27.927392961995793
42.65149840299273
47.20105353501276
52.18573272200592
69.59029746000306
58.178059666999616
67.04669021700101
65.99193224300689
84.94384207599796
100.03522219401202
104.99448512899107
102.68254196499765];
% Perform linear regression
coefficients = polyfit(images, times, 1);
coefficients_2 = polyfit(images, other_times, 1);
% Extract the slope (m) and intercept (b) of the best-fit line
m = coefficients(1); % Slope
b = coefficients(2); % Intercept
m1 = coefficients_2(1);
b1 = coefficients_2(2);
% Create a range of x-values for the best-fit line
x_range = min(images):100:max(images); % Adjust the step size as needed

% Calculate the corresponding y-values using the linear model
y_fit = m * x_range + b;
y_fit2 = m1 * x_range + b1;

% Plot the data and the best-fit line
figure;

% legend("")
hold on;
scatter(images, times, 'o', 'filled','b'); % Plot data points
scatter(images, other_times, 'x', 'r');
plot(x_range, y_fit2, 'r')
% plot(images, other_times)
plot(x_range, y_fit, 'b'); % Plot the best-fit line in red
hold off;
legend("Tree Approximation", "True Likelihood")
% Add labels and a legend
xlabel('Images');
ylabel('Runtime (seconds)');
title('Best Fit for Images vs. Times');

grid on;

% Display the equation of the best-fit line
fprintf('Best Fit Equation: y = %.4fx + %.4f\n', m, b);
