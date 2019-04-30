close all;
clear;
clc;

% Before training on real dataset, we test our algorithm on a simple dataset 
x = [0,0;2,2;2,0;3,0];
y = [0;0;1;1];
c = [1;1;1;1];
x_hom = [c x]; % homogeneous form

%% BFGS

[N, D] = size(x);
W = ones(D, 1);

yita = 0.1; % learning rate
iteration = 0;
epsilon = 1e-5;   
x_col_max = max(x, [], 1)
x = x ./ x_col_max
x = (x - 0.5)*2
I = diag(ones(D,1));

while (iteration == 0 || glist(iteration) > epsilon)
    y_ = sigmoid(x*W);
    dW = x'*(y_ - y);
    dW = dW / N;
    if iteration == 0
        D = diag(y_ .* (1 - y_));
        H = x' * D * x;
        H = inv(H);
        W_prior = W;
        W = W - H * dW;
        dW_prior = dW;
    else
        s = W - W_prior;
        ddW = dW - dW_prior;
        dv = dot(ddW, s);
        H = (I - (s*ddW')/dv) * H * (I - (ddW*s')/dv) + (s*s')/dv;
        W_prior = W;
        W = W - H * dW;
        dW_prior = dW;
    end
        
    iteration = iteration + 1;
    glist(iteration) = norm(dW);
end

%% plot ||\nabla g|| of simple dataset
semilogy(glist);
title('BFGS')
set(gca,'FontSize',15)
xlabel('iteration','FontSize',15)
ylabel('log ||\nabla g||', 'FontSize',15)

