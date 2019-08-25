%{
This function performs the predition for Immigrate(Iterative 
Max-Min Entropy Margin-Maximization with Interaction Terms) algorithm.
%}
function [class, prob] = predictImmigrate(w, train_xx, train_yy, newx, sig)
%{
INPUT:
w: weight matrix obtained from IMMIGRATE algorithm;
train_xx: model matrix of explanatory variables;
train_yy: label vector;
sig: sigma used in algorithm, default to be 1.
OUTPUT:
class: predicted class for newx
prob: predicted probabilities for newx
%}
if (nargin < 5), sig=1; end
N = length(train_yy);
label = unique(train_yy);
prob = zeros(size(newx,1),length(label));
for j = 1:length(label)
  for i = 1:size(newx,1)
      yyy = abs(train_yy-label(j));
      y = ones(N,1);
      y(yyy ~= 0) = 0;
      difference = abs(train_xx - repmat(newx(i,:),N,1)); 
      difference = sum((difference*w).*difference,2);
      tmp = y.*exp(-difference/sig); 
      s = sum(tmp);
      if s ~= 0 
          tmp = tmp/s;
      end
      prob(i,j) = difference.'*tmp;
  end
end

prob = prob./repmat(sum(prob,2),1,size(prob,2));
[~,pos_max] = min(prob.');
class= label(pos_max);


