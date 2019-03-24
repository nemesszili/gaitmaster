function avgDist = intraclass(xData,yData,labels)
% avgDist = intraclass(xData,yData,labels)
% assumes that each data point - a row in xData - belongs to a class and
% measures the intra-class sum-or squares there the possible labels are
% provided in LABELS

avgDist = 0;

for cClass = labels
    allClass = find(yData == cClass)';
    nClass   = length(allClass);
    for ii = allClass
        dII = xData(allClass,:) - repmat(xData(ii,:),[nClass,1]);
        avgDist = avgDist + sum(dII.*dII, 'all');
    end
end

avgDist = avgDist / size(xData,1);

end % function