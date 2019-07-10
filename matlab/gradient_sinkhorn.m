% compute the gradient of S_lambda (a,b) w.r.t. a
% Differential Properties of Sinkhorn Approximation for
% Learning with Wasserstein Distance
% a,b in probability simplex, M is the cost.
function [] = gradient_sinkhorn(a,b,M, lambda)

K = exp(-lambda.*M);
U = K.*M ;
[D,L,u,v]=sinkhornTransport(a,b,K,U,lambda);


end