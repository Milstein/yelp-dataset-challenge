function [T] = subset(T, b, u)
%% SUBSET Extracts a random sample of users and businesses.
%   Permutates the rows and columns, then cuts out the b-by-u matrix in the
%   top-left corner. Probably not uniform, but what the heck.
% PARAMETERS
%   T:    n-by-3 matrix, with the columns: biz | user | rating

    U = max(T(:,2));
    B = max(T(:,1));
    users = randsample(U, u);
    bizes = randsample(B, b);

    R = sparse(T(:,1), T(:,2), T(:,3));
    R = R(bizes, users);
    T = zeros(nnz(R),3);
    [T(:,1), T(:,2), T(:,3)] = find(R);
    
%     % HACK - hope I don't offend any statisticians
%     R = sparse(T(:,1), T(:,2), T(:,3));
%     R = R(randperm(size(R,1)),:);
%     R = R(:,randperm(size(R,2)));
%     R = R(1:b, 1:u);
%     T = zeros(nnz(R),3);
%     [T(:,1), T(:,2), T(:,3)] = find(R);
    
end

