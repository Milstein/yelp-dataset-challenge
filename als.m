function [F,RMSE,M,U,objs] = als(R,k,varargin)
%% ALS Alternating Least Squares with incomplete matrix R
%
%   [F,RMSE,M,U] = ALS(R,k) factors the nonnegative incomplete matrix R into
%   M (m-by-k) and U (k-by-u).
%
%   The M and U matrices are chosen to minimize the objective function
%   that is defined as
%       D = |M*U - R|^2 + lambda*(|M|^2 + |U|^2)
%
%   Note that the norm is only taken over the known ratings.
%
%   The factorization uses an iterative method starting with random initial
%   values for M and U.  Because the objective function often has local
%   minima, repeated factorizations may yield different M and U values.
%
% PARAMETERS
%   R:          m-by-u sparse matrix - zeros are treated as missing values
%               (should this function be used on some dataset that has 0 as a
%               valid value, offset the data before passing it in.) This is
%               done for memory optimization purposes.
%   k:          number of latent factors
% OPTIONS
%   Pass them as 'Name', 'Value'
%   'maxiters'   number of iterations (defaults to 17)
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
    
    [m,u] = size(R);
    
    % parse input options
    p = prepare_parser(m,u,k);
    parse(p,R,k,varargin{:});
    maxiters = p.Results.maxiters;
    lambda = p.Results.lambda;
    M0 = p.Results.M0;
    U0 = p.Results.U0;
    max_pred = p.Results.max;
    min_pred = p.Results.min;
    
    % calculate GammaU and GammaM
    [GammaU, GammaM] = calculate_gamma(R);

    % iterate till convergence
    objs = zeros(maxiters,1);
    for i=1:maxiters
        [M,U,~,D] = iterate(M0,U0,R,lambda,GammaU,GammaM);
        M0 = M;
        U0 = U;
        objs(i) = D;
        % disp(RMSE);
        disp(D);
    end
    
    F = @(biz, users, mu) ...
            bsxfun(@max, ...
                bsxfun(@min, M(biz,:)*U(:,users), max_pred), ...
                min_pred);
    RMSE = calculate_loss(F,R);

end

function [GammaU, GammaM] = calculate_gamma(R)
    [m,u] = size(R);
    uratings = zeros(1,u);
    mratings = zeros(1,m);
    parfor i=1:u
        uratings(i) = nnz(R(:,i));
    end
    parfor i=1:m
        mratings(i) = nnz(R(i,:));
    end
    GammaU = diag(uratings);
    GammaM = diag(mratings);
end

function [p] = prepare_parser(m, u, k)

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

function [M,U,RMSE,D] = iterate(M0,U0,R,lambda,GammaU,GammaM)
%% ITERATE ALS iteration
    
    u = size(R,2);
    [m,k] = size(M0);
    
    % hold M fixed, solve for U
    parfor i=1:u
        [bizes, ~, ratings] = find(R(:,i));
        if isempty(ratings)         % this user did not rate anything
            continue;
        end
        X = M0(bizes,:);           % relevant bizes
        y = ratings;
        % U0(:,i) = (lambda*sqrt(size(y,1))*eye(k) + X'*X)\(X'*y);
        U0(:,i) = (lambda*size(y,1)*eye(k) + X'*X)\(X'*y);
    end

    % hold U fixed, solve for M
    % sum_squares = zeros(1,u);
    parfor i=1:m
        [~, users, ratings] = find(R(i,:));
        if isempty(ratings)         % nobody rated this biz
            continue;
        end
        X = U0(:,users)';           % relevant users
        y = ratings';
        % M0(i,:) = (lambda*sqrt(size(y,1))*eye(k) + X'*X)\(X'*y);
        M0(i,:) = (lambda*size(y,1)*eye(k) + X'*X)\(X'*y);
        sum_squares(i) = norm(ratings - M0(i,:)*X').^2;
    end
    
    M = M0;
    U = U0;
    RMSE = [];
    D = sum(sum_squares) + lambda*(norm(GammaM*M)^2+norm(U*GammaU)^2);
    % RMSE = sqrt(sum(sum_squares)/nnz(R));

end

