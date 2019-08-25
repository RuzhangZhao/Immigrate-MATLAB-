%{
This function performs IMMIGRATE(Iterative Max-Min Entropy 
Margin-Maximization with Interaction Terms) algorithm.
%}
function [new_W, C, iter] = Immigrate(train_xx, train_yy, w0, removesmall, sig, max_iter, epsilon)
%{
INPUT:
train_xx: model matrix of explanatory variables;
train_yy: label vector;
w0: initial weight matrix;
removesmall: whether to remove features with small weights, 
default to be FALSE;
sig: sigma used in algorithm, default to be 1;
max_iter: maximum number of iteration;
epsilon: criterion for stopping iteration.
OUTPUT:
new_W: new weight matrix;
C: cost;
iter: number of iteration.
%}
p = size(train_xx,2);
if (nargin < 7), epsilon = 0.01; end
if (nargin < 6), max_iter = 10; end
if (nargin < 5), sig = 1; end
if (nargin < 4), removesmall = false; end
if nargin < 3
% random initialization of weight matrix
    A = rand(p);
    w0 = tril(A,-1)+triu(A',0);
    w0 = w0/sqrt(sum(w0(:).^2));
end
c0 = 0;
c_before = c0;
c_after = c0+1;
w_after = w0;
iter = 0;
if removesmall == false
% do not operate removesmall option here
    while (abs(c_before - c_after)>epsilon)&(iter< max_iter)
        w_before = w_after;
        c_before = c_after;
        [tmp, c_after] = oneImmigrate(train_xx,train_yy,w_before,sig);
        w_after = tmp;
        iter = iter + 1;
    end
else
% operate removesmall option here 
    while (abs(c_before - c_after)>epsilon)&(iter< max_iter)
        w_before = w_after;
        c_before = c_after;
        [tmp, c_after] = oneImmigrate(train_xx,train_yy,w_before,sig);
        w_after = tmp;
        w_after(w_after < 1/p) = 0;
        iter = iter + 1;
    end
end
new_W = w_after;
C = c_after;