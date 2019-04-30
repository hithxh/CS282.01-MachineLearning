close all;
clear;
clc;

% Before training on real dataset, we test our algorithm on a simple dataset 
x = [0,0;2,2;2,0;3,0];
y = [0;0;1;1];
c = [1;1;1;1];
x_hom = [c x]; % homogeneous form

%% Newton

[N, D] = size(x);
W = ones(D, 1);

yita = 0.1; % learning rate
iteration = 0;
epsilon = 1e-5;
    
while (iteration == 0 || glist(iteration) > epsilon)
    y_ = sigmoid(x*W);
    dW = x'*(y_ - y);
    dW = dW / N;
    D = diag(y_ .* (1 - y_));
    H = x' * D * x
    W = W - inv(H) * dW
    iteration = iteration + 1;
    glist(iteration) = norm(dW);
    glist(iteration)    
end

%% plot ||\nabla g|| of simple dataset
semilogy(glist);
title('Newton')
set(gca,'FontSize',15)
xlabel('iteration','FontSize',15)
ylabel('log ||\nabla g||', 'FontSize',15)

