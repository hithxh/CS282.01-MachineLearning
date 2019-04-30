close all;
clear;
clc;
load nbadata;

nbadatahom = [ones(size(nba_data,1), 1) nba_data(:,1:end-1)];
cham_label = nba_data(:,end);
nbadatahom_col_max = max(nbadatahom, [], 1)
nbadatahom = nbadatahom ./ nbadatahom_col_max
nbadatahom = (nbadatahom - 0.5)*2
[N, D] = size(nbadatahom);
W = ones(D, 1);
iteration = 0;
epsilon = 1e-3;  
I = diag(ones(D,1));

while (iteration == 0 || glist(iteration) > epsilon)
    y_ = sigmoid(nbadatahom*W);
    dW = nbadatahom'*(y_ - cham_label);
    dW = dW / N;
    if iteration == 0
        D = diag(y_ .* (1 - y_));
        H = nbadatahom' * D * nbadatahom;
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

score = nbadatahom * W
y_p = ones(size(nba_data,1), 1)
y_p(find(score<=0)) = 0
Prediction_accuracy = mean(cham_label == y_p)