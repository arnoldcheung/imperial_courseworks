function SVM()

    % Solve the quadratic optimisation problem. Estimate the labels for 
    % each of the test samples and report the accuracy of your trained SVM 
    % utilising the ground truth labels for the test data.
    clear; 
    clc; 
    close all;
    
    load('X.mat'); 
    load('l.mat');
    load('X_test.mat');
    load('l_test.mat');    
    
    N = size(X, 2); % N = number of samples
    St = (1/N) * (X * X'); % Construct St 
    
    Ky = (l * l') .* (X' * inv(St)' * X); % Construct Ky = [yiyjxiStxj]
    C = 1; % C the error allowance
    
    % Parameters for quadprog
    H = Ky; 
    f = -1 * ones(N, 1);
    A = [];
    b = [];
    Aeq = l';
    beq = 0;
    lb = zeros(N, 1);
    ub = C * ones(N, 1);
    
    a = quadprog(H,f,A,b,Aeq,beq,lb,ub); % Compute the Lagrangian multipliers
    
    w = inv(St) * (X .* l') * a; % Compute w by differentiating Lagrangian w.r.t. w
    
    ind = a > 0.0001; % Get index of all alphas > 0 (tolerance)
    
    b = mean(l(ind) - (w' * X(:, ind))'); % KKT conditions, g(x) = 0 when a > 0
    
    predictions = (w' * X_test + b)'; % Substitute X_test into equation to get prediction
    
    % Change numerical predictions to discrete labels
    label_predictions = predictions;
    label_predictions(label_predictions >= 1) = 1;
    label_predictions(label_predictions <= -1) = -1;
    label_predictions = round(label_predictions);
    
    correct = predictions .* l_test > 0; % Correct label if prediction * label > 1
  
    accuracy = size(correct, 1) / size(X_test, 2); % to be calculated
    fprintf('Accuracy on the test set is %3.2f\n', accuracy);
end