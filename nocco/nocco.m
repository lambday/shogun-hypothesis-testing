%
% NOCCO implementation
% Written (W) 2014 Soumyajit De
%

function [statistic] = nocco(X,Y,params);

% number of samples
assert(size(X,1)==size(Y,1));
m = size(X,1);

% The following if-end is taken from MATLAB code written by Arthur Gretton 07/12/08
% from mmdTestBoot.m

%Set kernel size to median distance between points in aggregate sample
if params.sig == -1
	Z = [X;Y];  %aggregate the sample
	size1=size(Z,1);
	if size1>100
		Zmed = Z(1:100,:);
		size1 = 100;
	else
		Zmed = Z;
	end
	G = sum((Zmed.*Zmed), 2);
	Q = repmat(G,1, size1);
	R = repmat(G', size1,1);
	dists = Q + R - 2 * Zmed * Zmed';
	dists = dists-tril(dists);
	dists=reshape(dists, size1^2, 1);
	params.sig = sqrt(0.5*median(dists(dists > 0)));  %rbf_dot has factor two in kernel
end

% compute the kernel
K = rbf_dot(X, X, params.sig);
L = rbf_dot(Y, Y, params.sig);

% center the kernel matrices
H = eye(m, m) - 1/m*ones(m, m);
Gx = H*K*H;
Gy = H*L*H;

% compute statistic estimate I^NOCCO as per eq(8) in paper
Rx = Gx * inv(Gx + m*params.eps*eye(m, m));
Ry = Gy * inv(Gy + m*params.eps*eye(m, m));

statistic = trace(Rx*Ry);
