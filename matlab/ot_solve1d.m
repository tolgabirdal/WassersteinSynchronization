function [d, sigma] = ot_solve1d(mu, v, p, N)

mu_cdf = cumsum(mu);
v_cdf = cumsum(v);

iFv = inverseFunc(Fv, 1.0);
iFv = inverseFunc(Fv, 1.0);


if p == 1
    d = sum(abs(mu_cdf - v_cdf));
elseif p == 2
    d = sqrt(sum((mu_cdf - v_cdf).^2 ));
else
    d = pow(sum(pow(abs(mu_cdf - v_cdf), p)), 1/p);
end


interpCDF = inverseFunc(iFf+iGf);
y = [0; diff(interpCDF)];

[~,sigmaf] = sort(f(:));
[~,sigmag] = sort(g(:));

% Compute the inverse permutation \(\sigma_f^{-1}\).
sigmafi = [];
sigmafi(sigmaf) = 1:n^2;

% Compute the optimal permutation \(\sigma^\star\).
sigma = sigmag(sigmafi);

end