function [Z, W] = whitening(Y, mode)
    % Whitens the data Y using sphering or pca
    R = (Y * Y') / size(Y, 2);
    [U, D, tmp] = svd(R);
    D = diag(D);
    if strcmp(mode, 'pca')
        W = diag(1. ./ sqrt(D)) * U';
        Z = W * Y;
    elseif strcmp(mode, 'sph')
        W = U *  diag(1. ./ sqrt(D)) * U';
        Z = W * Y;
    end
end