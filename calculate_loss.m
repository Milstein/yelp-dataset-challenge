function [loss] = calculate_loss(F, R)
%CALCULATE_LOSS Calculates RMSE over known ratings
%
% PARAMETERS
%   F: prediction function
%   R: sparse test matrix

    m = size(R,1);
    sum_squares = zeros(1,m);
    mu = mean(nonzeros(R));
    parfor i=1:m                    % over each movie
        [~, users, ratings] = find(R(i,:));
        if isempty(ratings)         % this user did not rate anything
            continue;
        end
        sum_squares(i) = norm(ratings - F(i,users,mu)).^2;
    end

    % RMSE
    loss = sqrt(sum(sum_squares)/nnz(R));

end