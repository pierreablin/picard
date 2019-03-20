function [Y, W] = picardo(X, m, maxiter, tol, lambda_min, ls_tries, verbose)
% Runs the Picard-O algorithm
%
% The algorithm is detailed in::
%
%   Pierre Ablin, Jean-François Cardoso, Alexandre Gramfort
%   Faster ICA under orthogonal constraint
%   ICASSP, 2018
%   https://arxiv.org/abs/1711.10873
%
% Parameters
% ----------
% X : array, shape (N, T)
%     Matrix containing the signals that have to be unmixed. N is the
%     number of signals, T is the number of samples. X has to be centered and
%     white.
%
% m : int
%     Size of L-BFGS's memory. Typical values for m are in the range 3-15
%
% maxiter : int
%     Maximal number of iterations for the algorithm
%
% tol : float
%     tolerance for the stopping criterion. Iterations stop when the norm
%     of the projected gradient gets smaller than tol.
%
% lambda_min : float
%     Constant used to regularize the Hessian approximation. The
%     eigenvalues of the approximation that are below lambda_min are
%     shifted to lambda_min.
%
% ls_tries : int
%     Number of tries allowed for the backtracking line-search. When that
%     number is exceeded, the direction is thrown away and the gradient
%     is used instead.
%
% whiten : bool
%     If true, the algorithm whitens the input signals. If False, the input
%     signals should already by white.
%
% verbose : boolean
%     If true, prints the informations about the algorithm.
%
% Returns
% -------
% Y : array, shape (N, T)
%     The estimated source matrix
%
% W : array, shape (N, N)
%     The estimated unmixing matrix, such that Y = WX.

% Authors: Pierre Ablin <pierre.ablin@inria.fr>
%          Alexandre Gramfort <alexandre.gramfort@inria.fr>
%          Jean-Francois Cardoso <cardoso@iap.fr>
%
% License: BSD (3-clause)

% Init
[N, T] = size(X);

W = eye(N);
Y = X;

s_list = {};
y_list = {};
r_list = {};
current_loss = Inf;
sign_change = false;

for n=1:maxiter
    % Compute the score function
    psiY = score(Y);
    psidY_mean = score_der(psiY);
    % Compute the relative gradient
    g = gradient(Y, psiY);
    % Compute the signs of the kurtosis
    K = psidY_mean - diag(g);
    signs = sign(K);
    if n > 1
        sign_change = any(signs ~= old_signs);
    end
    old_signs = signs;
    % Update the gradient
    g = diag(signs) * g;
    psidY_mean = psidY_mean .* signs;
    % Project
    G = (g - g') / 2.;
    % Stopping criterion
    gradient_norm = max(max(abs(G)));
    if gradient_norm < tol
        break
    end
    % Update the memory
    if n > 1
        s_list{end + 1} = direction;
        y = G - G_old;
        y_list{end + 1} = y;
        r_list{end + 1} = 1. / sum(sum(direction .* y));
        if length(s_list) > m
            s_list = s_list(2:end);
            y_list = y_list(2:end);
            r_list = r_list(2:end);
        end
    end
    G_old = G;
    % Flush the memory if there is a sign change.
    if sign_change
        current_loss = Inf;
        s_list = {};
        y_list = {};
        r_list = {};
    end
    % Compute the Hessian approximation and regularize
    h = proj_hessian_approx(Y, psidY_mean, g);
    h = regularize_hessian(h, lambda_min);
    % Find the L-BFGS direction
    direction = l_bfgs_direction(G, h, s_list, y_list, r_list);
    % Do a line_search in that direction :
    [converged, new_Y, new_loss, alpha] = line_search(Y, signs, direction, current_loss, ls_tries);
    % If the line search fails, restart in the gradient direction
    if ~converged
        direction = -G;
        s_list = {};
        y_list = {};
        r_list = {};
        [tmp, new_Y, new_loss, alpha] = line_search(Y, signs, direction, current_loss, ls_tries);
    end
    direction = alpha * direction;
    Y = new_Y;
    W = expm(direction) * W;
    current_loss = new_loss;
    if verbose
        fprintf('iteration %d, gradient norm = %.4g\n', n, gradient_norm)
    end
end

function [score] = score(Y)
    score = tanh(Y);
end

function [dscore] = score_der(psiY)
    dscore = -mean(psiY.^2, 2) + 1.;
end

function [grad] = gradient(Y, psiY)
    % Compute the gradient for the current signals
    T = size(Y, 2);
    grad = (psiY * Y') / T;
end

function [hess] = proj_hessian_approx(Y, psidY_mean, G)
    % Computes the projected Hessian approximation.
    N = size(Y, 1);
    diagonal = psidY_mean * ones(1, N);
    off_diag = diag(G);
    off_diag = repmat(off_diag, 1, N);
    hess = 0.5 * (diagonal + diagonal' - off_diag - off_diag');
end

function [h] = regularize_hessian(h, l)
    % Clips the eigenvalues of h to l
    h(h < l) = l;
end

function [direction] = solve_hessian(G, h)
    % Returns the inverse Hessian times G
    direction = G ./ h;
end

function [output] = loss(Y, signs)
    % Returns the loss function, evaluated for the current signals
    output = 0.;
    [N, T] = size(Y);
    for ii=1:N
        y = Y(ii, :);
        s = signs(ii, :);
        output = output + s * (sum(abs(y) + log1p(exp(-2. * abs(y))))) / T;
    end
end

function [direction] = l_bfgs_direction(G, h, s_list, y_list, r_list)
    q = G;
    a_list = {};
    for ii=1:length(s_list)
        s = s_list{end - ii + 1};
        y = y_list{end - ii + 1};
        r = r_list{end - ii + 1};
        alpha = r * sum(sum(s .* q));
        a_list{end + 1} = alpha;
        q = q - alpha * y;
    end
    z = solve_hessian(q, h);
    for ii=1:length(s_list)
        s = s_list{ii};
        y = y_list{ii};
        r = r_list{ii};
        alpha = a_list{end - ii + 1};
        beta = r * sum(sum(y .* z));
        z = z + (alpha - beta) * s;
    end
    direction = -z;
end

function [converged, Y_new, new_loss, alpha] = line_search(Y, signs, direction, current_loss, ls_tries)
    % Performs a backtracking line search, starting from Y and W, in the
    % direction direction.
    alpha = 1.;
    if current_loss == Inf
        current_loss = loss(Y, signs);
    end
    for ii=1:ls_tries
        Y_new = expm(alpha * direction) * Y;
        new_loss = loss(Y_new, signs);
        if new_loss < current_loss
            converged = true;
            return
        end
        alpha = alpha / 2.;
    end
    converged = false;
end

end
