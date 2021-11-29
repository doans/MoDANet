function [gradients,state,loss,lossLabels,lossAngles, dlY1, dlY2] = modelGradients(dlnet,dlX,Y1,Y2)

[dlY1,dlY2,state] = forward(dlnet,dlX,'Outputs',{'softmax_1','softmax_2'});

lossLabels = crossentropy(dlY1,Y1);
lossAngles = crossentropy(dlY2,Y2); 
loss = lossLabels + lossAngles;
% lossAngles = crossentropy(dlY2,Y2); 
% loss = lossLabels + lossAngles;
gradients = dlgradient(loss,dlnet.Learnables);

end