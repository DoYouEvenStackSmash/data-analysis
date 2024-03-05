% close all;
arr = [double(area_elem_indices);double(single_point_elem_indices)];
names = ["Patient","Greedy"]
figure;
for i=1:2
    idx = zeros
    idx = arr(i,:);
    sdy = [0:length(naive_matrix(:,1))-1];
    
    clear pts;
    pts = zeros(1,length(idx));
    pts(1,:) = idx;
    % pts(1,:) = [0:length(naive_matrix(:,1))-1];
    vals = zeros(2,length(idx));
    rm = real(naive_matrix).';
    rm(:,:) = sort(rm(:,:));
    % f1 = figure;
    subplot(1,2,i)
    % mesh(sorted_data);
    scatter3(sdy,pts,vals,"red",'x','LineWidth',1);
    % legend("A","B","C")
    hold on;
    % mesh(x, y);
    % stem3(sdy,zeros(length(sdy)),double(area_elem_indices),'.')
    % hold on;
    mesh(rm,"EdgeColor","interp","FaceAlpha",0.5);
    colorbar;   
    legend("Index",Location="south")
    title([names(i)+ ' Likelihood Ranks on Naive Matrix']);
    xlabel('Structure Index');
    ylabel('Image Index');
    zlabel("Likelihood");
    ylim([0, length(rm(:,1))]);
    xlim([0, length(rm(1,:))]);
    hold off;
    view([0,90]);
end