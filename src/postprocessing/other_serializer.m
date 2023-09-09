% Given data
images = [1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680];
times = [0.912749770010123, 2.989286013005767, 2.653218198000104, 3.677179874997819, 4.971076945003006, 4.609119443004602, 4.43832607899094, 5.015284392007743, 6.4422190339973895, 7.567061241003103, 7.886000758007867, 8.607719869993161, 9.072583952001878, 8.452264118997846];
other_times = [13.085189700999763,22.20253994000086,27.927392961995793,42.65149840299273,47.20105353501276,52.18573272200592,69.59029746000306,58.178059666999616,67.04669021700101,65.99193224300689,84.94384207599796,100.03522219401202,104.99448512899107,102.68254196499765];
disp(length(other_times));
disp(length(images));
% Combine the data into a matrix with three columns
data_matrix = [images', times', other_times'];

% Create column headers
column_headers = {'images', 'approx_likelihood_runtime', 'true_likelihood_runtime'};

% Create a table from the data and headers
data_table = array2table(data_matrix, 'VariableNames', column_headers);

% Save the table to a CSV file
writetable(data_table, 'output_data.csv');