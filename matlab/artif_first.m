%% script to test optimised projection

%% generating data

c = [ 5,0; ... % first center
      -2,26]; % second center
nData = 300;
yData = randi(nLabels,nData,1);

[nDim,nLabels]  = size(c);
cLabels = 1:nLabels;
xData = randn(nData,nDim) + c(yData,:);

% for proof -- visualising classes
cla
hist(yData,0.5:.5:nLabels);
axis tight

%% setting up the projection

% the number of components
K = 2;
V = randn(nDim,K);

% visualising the data
vis_class_data(xData, cLabels, yData,K);


%% performing the stochastic gradient adaptation
nSGD   = 10;
alpha  = 0.001;
nEpoch = 20;

for kk = 1:nEpoch
    ind1 = randperm(nData,nSGD);
    ind2 = randperm(nData,nSGD);

    x_sign_cov = zeros(nDim);
    for ii = ind1
        % setting up the labels
        eqV  = -ones(nSGD,1);
        eqV(yData(ii) == yData(ind2)) = nLabels;
        % computing weighted correlation
        diffData = xData(ind2,:) - repmat(xData(ii,:),[nSGD,1]);
        x_sign_cov = x_sign_cov + ...
            ( repmat(eqV,[1,nDim]) .*  diffData)' * ...
            diffData;
    end
   
    V = V - alpha * x_sign_cov * V / (nSGD*nSGD);
    % ortho-normalisation the COLUMNS of the projection matrix
    V = orthonorm(V')';

    % visualising the data
    vis_class_data(xData*V, cLabels, yData,K);
    pause  
end