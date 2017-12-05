function y = laplace_rnd(shape, mu, b)
% LAPRND generates random Laplacian numbers

if nargin < 2 || isempty(mu)
    mu = 0.;
end

if nargin < 3 || isempty(b)
    b = 1. / sqrt(2);
end

u = rand(shape) - 0.5;
y = mu - b * sign(u) .* log(1 - 2 * abs(u));

end
