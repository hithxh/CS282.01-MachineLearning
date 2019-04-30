% test example code for glasso, 
% Note that [m,n] = size(F), m >= n is well-posed
clc;
clear;

rng('default')
m = 60;
p = 200;
n = 150;


% parameters to choose  
lambda1 = 0.1;
lambda2 = 0.15;
lambda3 = 0.2;
lambda4 = 0.25;
lambda5 = 0.3;
lambda6 = 0.35;

x = randn(n,1);

x(1) = 1.53;

v = x + norm(x) *  [1;zeros(n-1,1)];
% F is householder matrix, transform x to a sparse vector
F = (eye(n) - 2*v*v'/(v'*v)); 

% measure matrix
A = randn(m,n); 

 % observation with noise
b = A * x + 0.01 * randn(m,1); 

x_hat = glasso( A, b, F, lambda3, 1e-8 ); % choose the best lambda 

z = F * x;
z_hat = F * x_hat;
relative_error_x = norm(x_hat-x)/norm(x)
relative_error_z = norm(z_hat-z)/norm(z)






