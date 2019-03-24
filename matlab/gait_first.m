%% script to test the gait analysis NEMES

%% importing the data

gait_0= csvread('gait.csv');
% separating X and Y
xData = gait_0(:,1:end-1);
yData = gait_0(:,end);
[nData, nDim] = size(xData);

% finding the number of classes
cLabels = unique(yData)';
nLabels = length(cLabels);

% for proof -- visualising classes
cla
hist(yData,1:.5:22);
axis tight

%% setting up the projection

% the number of components
K = 2;
V = randn(nDim,K);

% visualising the data
vis_class_data(xData*V, cLabels, yData,K);


%% performing the stochastic gradient adaptation
nSGD   = 400;
alpha  = 0.001;
nEpoch = 50000;

for kk = 1:nEpoch
    ind1 = randperm(nData,nSGD);
    ind2 = randperm(nData,nSGD);

    x_sign_cov = zeros(nDim);
    for ii = ind1
        eqV  = -ones(nSGD,1);
        eqV(yData(ii) == yData(ind2)) = nLabels^2;
        % computing weighted correlation
        diffData = xData(ind2,:) - repmat(xData(ii,:),[nSGD,1]);
        x_sign_cov = x_sign_cov + ...
            ( repmat(eqV,[1,nDim]) .*  diffData)' * ...
            diffData;
    end
   
    V = V - alpha * x_sign_cov * V / (nSGD);
    % ortho-normalisation the COLUMNS of the projection matrix
    V = orthonorm(V);

    if ~mod(kk,100)
        % visualising the data
        vis_class_data(xData*V, cLabels, yData,K);
        pause(.1);
        drawnow;
        fprintf('Dist: %5.4f\n', intraclass(xData*V,yData,cLabels) );
    end
    
end