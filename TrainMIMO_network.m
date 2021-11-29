% clear all
% close all
%% load data
% reset(gpuDevice)
% Before running the code you should download dataset from this link: https://ieee-dataport.org/documents/multi-task-learning-dataset-automatic-modulation-classification-and-doa-estimation
load('myData_info.mat');
N = numel(T.Files);
indices = randperm(N);
indices_train = indices(1:round(0.8*N));
indices_val = indices(round(0.8*N)+1:end);

ds_Train = T.Files(indices_train);
response1_Train = T.modulation(indices_train);
response2_Train = T.DOA(indices_train);

ds_Val = T.Files(indices_val);
response1_Val = T.modulation(indices_val);
response2_Val = T.DOA(indices_val);
clear T
classes = sort(categorical({'16QAM','64QAM',...
    'PSK','QPSK', '8PSK', ...
    'QFSK','8FSK',...
    '16APSK', '4PAM'...
    'LFM', 'DSB-SC', 'SSB-SC'}));
az = sort(-60:60);
miniBatchSize = 256;
numClasses = numel(classes);
InputSize = [1024, 2, 5];
nofilters = 64;
filterSize = [9,1];
%% Define Deep Learning Model
% Mdl_1_DifFiltersSize;
lgraph = getMdl(InputSize,nofilters,filterSize);
dlnet = dlnetwork(lgraph);

%% Training model
% Specify Training Options
numEpochs = 10;

initialLearnRate = 0.01;
decay = 0.01;
momentum = 0.9;

plots = "training-progress";
executionEnvironment = "gpu";

if plots == "training-progress"
    fig = figure('Name',plots,'NumberTitle','off');
    fig.Position = [100, 100, 1024, 512];
    subplot(2,2,1)
    lineLossTrain = animatedline('Color','blue');
    EpochLossTrain = animatedline('Color','black','Marker','o','MarkerFaceColor','black');
    lineLossLabels = animatedline('Color','g');
    lineLossAngles = animatedline('Color','r');
    
    %     ylim([0 inf])
    xlabel("Epoch")
    ylabel("Training Loss")
    grid on
    subplot(2,2,2)
    lineAccuracyTest = animatedline('Color','g');
    EpochAccuracyTest = animatedline('Color','black','Marker','o','MarkerFaceColor','black');
    ylim([0 100])
    xlabel("Epoch")
    ylabel("Test Modulation Class. Accuracy (%)")
    grid on
    subplot(2,2,3)
    lineRMSETest = animatedline('Color','b');
    EpochRMSETest = animatedline('Color','black','Marker','o','MarkerFaceColor','black');
    ylim([0 100])
    %     ylim([0 inf])
    xlabel("Epoch")
    ylabel("Test Angle Class. Accuracy (%)")
    grid on
end
velocity = [];


numObservations = numel(ds_Train);
% Train the model.
iteration = 0;
start = tic;


dlYPred1 = [];
dlYPred2 = [];
for ii = 1:miniBatchSize:numel(ds_Val)
    XVal = ds_Val(ii:min(ii+miniBatchSize-1,numel(ds_Val)));
    ModTypes = response1_Val(ii:min(ii+miniBatchSize-1,numel(ds_Val)));
    Angles = response2_Val(ii:min(ii+miniBatchSize-1,numel(ds_Val)));
    X = ReadData(XVal);
    
    % Convert mini-batch of data to dlarray.
    dlX = dlarray(X,'SSCB');
    if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
        dlXTest = gpuArray(dlX);
    end
    [dlY1,dlY2] = modelPredictions(dlnet,dlXTest,miniBatchSize,numClasses);
    dlYPred1 = [dlYPred1 dlY1];
    dlYPred2 = [dlYPred2 dlY2];
end
[~,idx] = max(extractdata(dlYPred1),[],1);
YPred1 = classes(idx);
YPred1 = categorical(YPred1);
accuracy1Val = mean(YPred1 == response1_Val)*100;
disp("Validation Modulation accuracy: " + accuracy1Val + "%")

[~,idx] = max(extractdata(dlYPred2),[],1);
YPred2 = az(idx);
accuracy2Val = rms(YPred2 - response2_Val);
disp("Validation Angle accuracy: " + accuracy2Val + "degrees")

%     addpoints(EpochLossTrain,epoch,double(gather(extractdata(loss))))
addpoints(EpochAccuracyTest,0,accuracy1Val)
addpoints(EpochRMSETest,0,accuracy2Val)
drawnow


% Loop over epochs.
for epoch = 3:numEpochs
    %     shuffle data
    indices = randperm(numObservations);
    ds_Train = ds_Train(indices);
    response1_Train = response1_Train(indices);
    response2_Train = response2_Train(indices);
    
    indices = randperm(numel(ds_Val));
    ds_Val = ds_Val(indices);
    response1_Val = response1_Val(indices);
    response2_Val = response2_Val(indices);
    
    
    numIterationsPerEpoch = ceil(numObservations./miniBatchSize);
    % Loop over mini-batches.
    for ii = 1:numIterationsPerEpoch
        iteration = iteration + 1;
        
        % Read mini-batch of data and convert the labels to dummy
        % variables.
        %         idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
        XTrain = ds_Train((ii-1)*miniBatchSize+1:min(ii*miniBatchSize,numObservations));
        ModTypes = response1_Train((ii-1)*miniBatchSize+1:min(ii*miniBatchSize,numObservations));
        Angles = response2_Train((ii-1)*miniBatchSize+1:min(ii*miniBatchSize,numObservations));
        X = ReadData(XTrain);
        %         Y = single(Response(:,idx));
        Y1 = zeros(numClasses, numel(XTrain), 'single');
        for c = 1:numClasses
            Y1(c,ModTypes==classes(c)) = 1;
        end
        Y2 = zeros(121, numel(XTrain), 'single');
        for c = 1:121
            Y2(c,Angles==az(c)) = 1;
        end
        %         Y2 = single(Angles);
        % Convert mini-batch of data to dlarray.
        dlX = dlarray(X,'SSCB');
        
        % If training on a GPU, then convert data to gpuArray.
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            dlX = gpuArray(dlX);
        end
        
        % Evaluate the model gradients, state, and loss using dlfeval and the
        % modelGradients function and update the network state.
        [gradients,state,loss,lossModulation,lossAngle,dlY1, dlY2] = dlfeval(@modelGradients,dlnet,dlX,Y1,Y2);
        dlnet.State = state;
        
        % Determine learning rate for time-based decay learning rate schedule.
        learnRate = initialLearnRate/(1 + decay*iteration);
        
        % Update the network parameters using the SGDM optimizer.
        [dlnet, velocity] = sgdmupdate(dlnet, gradients, velocity, learnRate, momentum);
        [~,idx] = max(extractdata(dlY1),[],1);
        YPred1 = classes(idx);
        YPred1 = categorical(YPred1);
        accuracy1 = mean(YPred1 == ModTypes)*100;
        %         disp("Modulation accuracy: " + accuracy1 + "%")
        
        [~,idx] = max(extractdata(dlY2),[],1);
        YPred2 = az(idx);
        accuracy2 = rms(YPred2 - Angles);
        %         disp("Angle accuracy: " + accuracy2 + "%")
        %         angleRMSE = sqrt(mean((extractdata(dlYPred2) - YTest2).^2))
        
        % Display the training progress.
        if plots == "training-progress"
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            subplot(2,2,1)
            title("Epoch: " + epoch + ", Elapsed: " + string(D) + ...
                ", learnRate: " + string(learnRate))
            addpoints(lineLossTrain,iteration/numIterationsPerEpoch,double(gather(extractdata(loss))))
            addpoints(lineLossLabels,iteration/numIterationsPerEpoch,double(gather(extractdata(lossModulation))))
            addpoints(lineLossAngles,iteration/numIterationsPerEpoch,double(gather(extractdata(lossAngle))))
            addpoints(lineAccuracyTest,iteration/numIterationsPerEpoch,accuracy1)
            addpoints(lineRMSETest,iteration/numIterationsPerEpoch,accuracy2)
            drawnow
        end
        
    end
    
    dlYPred1 = [];
    dlYPred2 = [];
    for ii = 1:miniBatchSize:numel(ds_Val)
        XVal = ds_Val(ii:min(ii+miniBatchSize-1,numel(ds_Val)));
        ModTypes = response1_Val(ii:min(ii+miniBatchSize-1,numel(ds_Val)));
        Angles = response2_Val(ii:min(ii+miniBatchSize-1,numel(ds_Val)));
        X = ReadData(XVal);
        
        % Convert mini-batch of data to dlarray.
        dlX = dlarray(X,'SSCB');
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            dlXTest = gpuArray(dlX);
        end
        [dlY1,dlY2] = modelPredictions(dlnet,dlXTest,miniBatchSize,numClasses);
        dlYPred1 = [dlYPred1 dlY1];
        dlYPred2 = [dlYPred2 dlY2];
    end
    [~,idx] = max(extractdata(dlYPred1),[],1);
    YPred1 = classes(idx);
    YPred1 = categorical(YPred1);
    accuracy1Val = mean(YPred1 == response1_Val)*100;
    disp("Validation Modulation accuracy: " + accuracy1Val + "%")
    
    [~,idx] = max(extractdata(dlYPred2),[],1);
    YPred2 = az(idx);
    accuracy2Val = rms(YPred2 - response2_Val);
    disp("Validation Angle accuracy: " + accuracy2Val + "degrees")
    
    %     addpoints(EpochLossTrain,epoch,double(gather(extractdata(loss))))
    addpoints(EpochAccuracyTest,epoch,accuracy1Val)
    addpoints(EpochRMSETest,epoch,accuracy2Val)
    drawnow
    checkpoints(dlnet,epoch)
end


conf = confusionmat(categorical(response1_Val),YPred1');
confmat = conf(25:end,25:end);

figure
cm = confusionchart(conf,classes);
cm.Title = 'Confusion Matrix';
cm.RowSummary = 'row-normalized';
% cm.FontName = 'Times New Roman';
cm.FontColor =[0 0 0];
cm.FontSize = 12;
colorbar = [0 0 1];
cm.DiagonalColor = colorbar;
cm.Parent.Position = [cm.Parent.Position(1:2) 740 424];
cm.OffDiagonalColor = colorbar;
