function U = wPCA(data)
    data=data'; % Transpose the data into features x samples        
    data = data - mean(data, 2); % Centralise the data
    C = data' * data; % Calculate C (inner product instead of outer product)
    [V, S, ~] = svd(C); % SVD for C to get eigenvalues and eigen vectors
    U=data*V*S^(-1); % Transform back to eigenvectors of the covariance matrix
end