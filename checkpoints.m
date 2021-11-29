function checkpoints(dlnet,epoch)
checkpointPath = fullfile("checkpoints");
if ~exist(checkpointPath,"dir")
    mkdir(checkpointPath)
end
if ~isempty(checkpointPath)
    D = datestr(now,'yyyy_mm_dd__HH_MM_SS');
    filename = fullfile(checkpointPath,"dlnet_checkpoint__Epoch" + epoch + "__" + D + ".mat");
    save(filename,"dlnet")
end
