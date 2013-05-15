function [loss] = cv(algo,T,k,varargin)
%% CV CROSS-VALIDATION
% Calculates mean RMS over all folds
% PARAMETERS
%   algo: algorithm to use
%   T:    n-by-3 matrix, with the columns: biz | user | rating
%   k:    number of folds

    n = length(T);
    b = max(T(:, 1));
    u = max(T(:, 2));
    indices = crossvalind('Kfold',n,k);
    test_loss = zeros(1,k);

    % matlabpool open
    for i = 1:k             % for each fold
        test = (indices == i); train = ~test;
        train_matrix = sparse(T(train,1),T(train,2),T(train,3),b,u);
        % run algo on train data
        [F,~] = algo(train_matrix,varargin{:});
        % calculate loss on test data
        test_matrix = sparse(T(test,1),T(test,2),T(test,3),b,u);
        test_loss(i) = calculate_loss(F, test_matrix);
    end
    % matlabpool close

    loss = mean(test_loss);

end

% EXAMPLE
%  cv(@als, TSub, 5, 5, 'min', 1, 'max', 5, 'lambda', 0.1)