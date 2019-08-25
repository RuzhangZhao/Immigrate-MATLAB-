%{
This function performs IMMIGRATE(Iterative Max-Min Entropy 
Margin-Maximization with Interaction Terms) algorithm for one loop.
%}
function [new_W, C] = oneImmigrate(train_xx,train_yy,W,sig)
%{
INPUT:
train_xx: model matrix of explanatory variables, 
where N*p: N is sample size, p is num of features;
train_yy: label vector;
W: initial weight matrix;
sig: sigma used in algorithm, default to be 1.
OUTPUT:
new_W: new weight matrix after one loop;
C: cost after one loop.
%}
if (nargin < 4), sig=1; end
[N,p] = size(train_xx);
entropy = 0;
MM = zeros(p);
% update of probability coefficients(alpha, beta)
for i = 1:N
    yyy = abs(train_yy - train_yy(i));
    yy = ones(1,N);
    yy(yyy ~= 0) = 0;
    difference = abs(train_xx - repmat(train_xx(i,:),N,1));
    tmp = exp(-sum((difference*W).*difference,2)/sig);
    tmp_0 = yy.'.*tmp;
    tmp_0(i) = 0;
    s0 = sum(tmp_0);
    if s0 ~= 0
        tmp_0 = tmp_0/s0;
    end
    tmp_1 = (1-yy).'.*tmp;
    s1 = sum(tmp_1);
    if s1 ~= 0
        tmp_1 = tmp_1/s1;
    end
    tmp_0(i) = 1;
    entropy = entropy + sum((tmp_0 - tmp_1).*log(abs(tmp_0 - tmp_1)));
    
    tmp_0(i) = 0;
    MM = MM + difference.'* bsxfun(@times,difference,(tmp_0-tmp_1));
end
% update of weight matrix W
[eigvect, eigvalue] = eig(-MM);
[eigvalue,index] = sort(diag(eigvalue),'descend');
eigvect = eigvect(:,index);
eigvalue = max(eigvalue,0);

if sum(eigvalue)>0
    eigvalue = eigvalue/sum(eigvalue);
end

new_W = eigvect*(bsxfun(@times,eigvect.',eigvalue));
new_W = max(new_W,0);
new_W = new_W/sqrt(sum(new_W(:).^2));
% calculation of cost value
C = sum(sum((eigvect.'*MM).*eigvect.')) + sig*entropy;
