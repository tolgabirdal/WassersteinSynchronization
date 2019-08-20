function B = bingham_fit(X)
n = size(X,1);
S = X'*X/n;
B = bingham_fit_scatter(S);
