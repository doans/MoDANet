function [dlYPred1,dlYPred2] = modelPredictions(dlnet,dlX,miniBatchSize,numClasses)

numObservations = size(dlX,4);
numIterations = ceil(numObservations / miniBatchSize);

% numClasses = dlnet.Layers(60).OutputSize;
dlYPred1 = zeros(numClasses,numObservations,'like',dlX);
dlYPred2 = zeros(121,numObservations,'like',dlX);
for i = 1:numIterations
    idx = (i-1)*miniBatchSize+1:min(i*miniBatchSize,numObservations);
    
    [dlYPred1(:,idx),dlYPred2(:,idx)] = predict(dlnet,dlX(:,:,:,idx),'Outputs',{'softmax_1','softmax_2'});
end

end