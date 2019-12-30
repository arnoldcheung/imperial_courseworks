clear; 
clc; 
close all;

warning('off','all');

load idx;
load fea;
load gnd;

error = [];

for i = 1:20
    fprintf('Processing batch %d...\n', i);
    indices = idx(i, :);
    
    test_idx = 1:size(fea, 1);
    test_idx(indices) = [];
    
    fea_train = fea(indices, :);
    gnd_train = gnd(indices); 
    [gnd_train, ind] = sort(gnd_train, 'ascend');
    fea_train = fea_train(ind, :);

    fea_test = fea(test_idx, :);
    gnd_test = gnd(test_idx);

    U = LDA(fea_train, gnd_train);  

    reduced_train_fea = fea_train*U;
    reduced_test_fea = fea_test*U;

    mg = mean(reduced_train_fea, 1);
    reduced_train_fea = reduced_train_fea - repmat(mg, size(reduced_train_fea,1), 1);
    reduced_test_fea = reduced_test_fea - repmat(mg, size(reduced_test_fea,1), 1);

    len = 1:size(reduced_test_fea,2);
    correct = zeros(1, length(1:size(reduced_test_fea,2)));
    
    for j = 1:length(len) 
        kept_test_fea = reduced_test_fea(:, 1:len(j));
        kept_train_fea = reduced_train_fea(:, 1:len(j));
        knn_model = fitcknn(kept_train_fea, gnd_train, 'Distance', 'cosine', 'NumNeighbors', 1);
        class = predict(knn_model, kept_test_fea);
        correct(j) = length(find(class-gnd_test == 0));
    end

    correct = correct./length(gnd_test);
    error = [error; 1-correct];
end

figLDA = figure;
axis1 = axes('Parent', figLDA);
hold(axis1, 'on');
% plot the error
plot(mean(error,1)); 
% create xlabel
xlabel('components');
% create title
title('LDA, PIE DB');
% create ylabel
ylabel('error rate');