function W = LDA(data, labels)

    data = data'; % Transpose the data into features x samples 
    data = data - mean(data, 2); % Centralise the data

    [~, ~, ic] = unique(labels, 'stable');
    counts = accumarray(ic, 1); % count number of samples in each class
    
    Es = {};
    for c = 1:length(counts)
        E = 1/counts(c) * ones(counts(c), counts(c)); % Compute E for each class
        Es{c} = E; % append blocks to the list of E matrices
    end    
    M = blkdiag(Es{:}); % Block diagonal matrix 
    
    % Finding U for W = U Q that transforms contraint into Q'Q = I
    % U' Sw U = I 
    % Uw' L Uw = Sw
    % -> L = Uw' Sw Uw where Uw = eigenvectos, L = eigenvalues
    % -> L^-0.5 Uw' Sw Uw L^-0.5 = I
    % L^-0.5 L L^-0.5 = I
    % Choose U = Uw L^-0.5
    
    C = ((eye(size(M)) - M) * data') * (data * (eye(size(M)) - M)); % Inner instead of outer product
    [Vw, Lw, ~] = svd(C); % SVD on C to get eigenvalues and eigenvectors
    
    evs = diag(Lw); % get eigenvalues
    
    tolerance = max(diag(Lw)) * 0.00001; % remove values that are too small compared to largest eigenvalue
    Vw(:, evs < tolerance) = []; % remove eigenvectors corresponding to small eigenvalues
    Lw(:, evs < tolerance) = []; % remove columns with small eigenvalues
    Lw(evs < tolerance, :) = []; % remove rows with small eigenvalues
   
    Uw = data * (eye(size(M)) - M) * Vw * Lw^(-0.5); % Eigenvectors of Sw
    U = Uw * Lw^(-0.5); % Choose U = Uw L^-0.5 
    
    X_tilde_b = U' * data * M;
        
    [Q, L, ~] = svd(X_tilde_b * X_tilde_b');
    
    W = U * Q;   
end