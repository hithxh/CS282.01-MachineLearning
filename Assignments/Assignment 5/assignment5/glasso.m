function [ x, iter ] = glasso( A, b, F, lambda, tol, maxiter )
%GLASSO  generalized Lasso problem
% 
%  x = GLASSO(A, b, F, lambda) returns the solution to generalized Lasso
%  problem
%     
%           arg min_x  0.5*||Ax-b||_2^2 + lambda||Fx||_1
%  which is equivalent to 
%           min_{x,z} 0.5*||Ax-b||_2^2 + lambda||z||_1
%             s.t.             Fx - z = 0
%
% input:
%       A           a matrix
%       b           a vector
%       F           a matrix
%       lambda      paramter, a scalar
%       tol         tolerance, default is 1e-6
%       maxiter     maxiteration, default is 500
%
% output:
%       x           the sloution to generalized Lasso problem, a vector
%       iter        iteration
%   
% Kejun Tang
% Last modified 03/11/2018

if nargin < 5
    tol = 1e-6;
end

if nargin < 6
    maxiter = 500;   
end

if length(b) ~= size(A, 1)
    error('matrix dimension (A and b) must agree!');
end

n = size(A, 2);
x0 = rand(n, 1);
%x0 = zeros(n, 1); % initialization for solution x
z = F * x0; % initialization for solution z

y = zeros(size(F,1),1); % Lagrange multiplier
rho = 0.5; % augmented Lagrange multiplier
r = 1.12; % ratio for rho
converged = 0;
iter = 0; % iteration

while ~converged
    
    iter = iter + 1;
    % ADMM
    % update x, write your formulation
    x1 = (inv(A'*A + rho * F'*F))*(A'*b + rho * F'*(z) - F'*y);
    % update z, write your formulation 
    z = soft((F*x1 + y/rho), (lambda/rho));
    
    y = y + rho * (F*x1 - z);
    
    rho = rho * r; % update rho
    converged = ((norm(x1-x0) <= tol) && (norm(F*x1-z) <= tol));
    
    fprintf('iteration: %d, augment Lagrange multiplier rho: %6.2f\n', iter, rho);
    fprintf('stopping criteron: deltaX%8.6f, constraint %8.6f\n', norm(x1-x0), norm(F*x1-z));
    fprintf('=================================================\n');
    
    if (iter >= maxiter) && ~converged
        fprintf('maximum iteration reached, but gLasso not converged!\n');
        break
    end
    
    x0 = x1;
    
end

x = x1;




end

