function [F,RMSE,M,U,MB,UB] = als_bias(R,k,varargin)
%% ALS Alternating Least Squares with incomplete matrix R
%
%   [F,RMSE,M,U,MB,UB] = ALS(R,k) factors the nonnegative incomplete matrix
%   R into factor matrices M (m-by-k) and U (k-by-u) and bias vectors MB and
%   UB.
%
%   The M and U matrices are chosen to minimize the objective function
%   that is defined as
%       D = |R - R_HAT|^2 + lambda*(|M|^2+|U|^2+|MB|^2+|UB|^2)
%   where each rating r(i,j) in R_HAT is calculated by
%       r(i,j) = M(i,:) * U(:,j) + MB(i) + UB(j)
%
%   Note that the norm is only taken over the known ratings.
%
%   The factorization uses an iterative method starting with random initial
%   values for M and U.  Because the objective function often has local
%   minima, repeated factorizations may yield different M and U values.
%
% PARAMETERS
%   R:          m-by-u sparse matrix - zeros are treated as missing values
%               (should this function be used on some dataset that has 0 as
%               a valid value, offset the data before passing it in.) This
%               is done for memory optimization purposes.
%   k:          number of latent factors
% OPTIONS
%   Pass them as 'Name', 'Value'
%   'maxiter'    number of iterations (defaults to 17)
%   'lambda'     regularization parameter (defaults to 0.01)
%   'M0'         initial values for M (defauls to random values in [0,1])
%   'U0'         initial values for U (defauls to random values in [0,1])
%   'max'        max prediction value
%   'min'        min prediction value
%
% RETURNS
%   F:           prediction function
%   RMSE:        root mean squared error over known ratings
%   M:           m-by-k factor matrix
%   U:           k-by-u factor matrix
%   MB:          bias vectors for bizes
%   UB:          bias vectors for users
    
    [m,u] = size(R);
    mu = mean(nonzeros(R));
    
    % parse input options
    p = prepare_parser(m,u,k);
    parse(p,R,k,varargin{:});
    maxiters = p.Results.maxiters;
    lambda = p.Results.lambda;
    M0 = p.Results.M0;
    U0 = p.Results.U0;
    max_pred = p.Results.max;
    min_pred = p.Results.min;

    % center R around its mean
    R_c = spfun(@(x) x - mu, R); % WARNING: mu might be a whole number ...
    
    % calculate GammaU and GammaM
    [GammaU, GammaM] = calculate_gamma(R);
    errors = zeros(1,maxiters);
    
    % iterate till convergence
    for i=1:maxiters
        [M,U,MB,UB,RMSE,~] = iterate(M0,U0,R_c,lambda,GammaU,GammaM);
        M0 = M;
        U0 = U;
        errors(i) = RMSE;
        % disp(D);
    end
    
    % prediction function
    F = @(biz, users, test_mu) ...
            bsxfun(@max, ...
                bsxfun(@min, ...
                    (M(biz,:)*U(:,users) + UB(users))+MB(biz)+mu, ...
                max_pred), ...          % upper bound
            min_pred);                  % lower bound
    RMSE = calculate_loss(F,R);

end

function [GammaU, GammaM] = calculate_gamma(R)
    [m,u] = size(R);
    uratings = zeros(1,u);
    mratings = zeros(1,m);
    parfor i=1:m
        mratings(i) = sqrt(nnz(R(i,:)));
    end
    parfor i=1:u
        uratings(i) = sqrt(nnz(R(:,i)));
    end
    GammaU = diag(uratings);
    GammaM = diag(mratings);
end

function [p] = prepare_parser(m, u, k)
%% PREPARE_PARSER

    p = inputParser;
    addRequired(p,'ratings',@ismatrix);
    addRequired(p,'features',@isnumeric);
    addParamValue(p,'maxiters',17,@isnumeric);
    addParamValue(p,'lambda',0.01,@isnumeric);
    addParamValue(p,'M0',rand(m,k));        
    addParamValue(p,'U0',rand(k,u));
    addParamValue(p,'max',inf);
    addParamValue(p,'min',-inf);
    
end

function [M,U,MB,UB,RMSE,D] = iterate(M0,U0,R,lambda,GammaU,GammaM)
%% ITERATE ALS iteration
%
% PARAMETERS:
%   R:      sparse matrix R centered around its mean
    
    u = size(R,2);
    [m,k] = size(M0);
    
    MB = zeros(1,m);
    UB = zeros(1,u);

    % hold U fixed, solve for M
    parfor i=1:m
        [~, users, ratings] = find(R(i,:));
        if isempty(ratings)          % nobody rated this business
            continue;
        end
        X = [ones(size(users,2),1) U0(:,users)']; % relevant users
        y = ratings';
        % regularized linear regression
        factor = (lambda*size(y,1)*eye(k+1) + X'*X)\(X'*y);
        MB(i) = factor(1);
        M0(i,:) = factor(2:k+1);
    end
    
    % hold M fixed, solve for U
    parfor i=1:u
        [bizes, ~, ratings] = find(R(:,i));
        if isempty(ratings)          % this user did not rate anything
            continue;
        end
        X = [ones(size(bizes,1),1) M0(bizes,:)]; % relevant businesses
        y = ratings;
        % regularized linear regression
        factor = (lambda*size(y,1)*eye(k+1) + X'*X)\(X'*y);
        UB(i) = factor(1);
        U0(:,i) = factor(2:k+1);
        % calculate sum of squares
        sum_squares(i) = norm(ratings - X*factor - MB(bizes)').^2;
    end
    
    M = M0;
    U = U0;
    D = [];
    % D = sum(sum_squares)+lambda*(...
    %        norm(GammaM*M)^2+...
    %        norm(U*GammaU)^2+...
    %        norm(MB*GammaM)^2+...
    %        norm(UB*GammaU)^2);
    RMSE = sqrt(sum(sum_squares)/nnz(R));


end

