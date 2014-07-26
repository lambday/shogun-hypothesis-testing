%
% m - number of samples
% dim - dimension
% eps - epsilon_n, regularization constant
% alpha - test power
% n - number of null samples
function [p_value] = nocco_test(m,dim,eps,alpha,n);

X = normrnd(0,1,m,dim);
Y = repmat((1:m)'./m,1,dim);

params.('eps') = eps;
params.('sig') = -1;

XY = [X;Y];

statistic = nocco(X, Y, params);
null_samples = zeros(n,1);

for i = 1:n
	[notUsed,indsShuff] = sort(rand(2*m,1));
	XYShuff = XY(indsShuff, 1:dim);
	XShuff = XYShuff(1:m, 1:dim);
	YShuff = XYShuff(m+1:2*m, 1:dim);
	null_samples(i) = nocco(XShuff, YShuff, params);
end

null_samples = sort(null_samples);
p_value = 1-find(null_samples>=statistic)(1)/n;
