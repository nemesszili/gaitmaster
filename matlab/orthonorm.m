function M = orthonorm(M)
% orthonormalising a matrix M by using Gauss-Jordan-like elimination
% NewM = orthonorm(M) makes the ROWS of the matrix M orthonormal.
%
% If there are insufficient indepentent components, the LAST ROW is only
% normalised.

%% 
[~,nComp] = size(M);
thrVal = 1e6*eps;

for ii=1:nComp
    tDir = M(:,ii) - M(:,1:(ii-1))*(M(:,1:(ii-1))'*M(:,ii));
    nDir = norm(tDir);
    if nDir > thrVal
        M(:,ii) = tDir / nDir;
    else
        M(:,ii) = M(:,ii) / ( thrVal + norm(M(:,ii)) );
    end
end
