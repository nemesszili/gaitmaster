function vis_class_data(xData, classes, yData, nAttr)
% vis_class_data(xData, classes, yData, nAttr) -- visualises the data xData. It 
% assumes that the first two arguments are displayed.
%
% if NATTR > 2 then 3D plot is generated. Default is 2.

if nargin<4
    nAttr = 2;
end

%
cla; hold on;

for iC = classes
    iData = find(yData==iC);
    if nAttr == 2
        plot(xData(iData,1),xData(iData,2), '*')
    else
        plot3(xData(iData,1),xData(iData,2),xData(iData,3), '.')
    end
end

axis tight
end % function
