function X = ReadData(XTrain)
N = numel(XTrain);
a = load(XTrain{1},'data');
[h,w,c] = size(a.data);
X = zeros(h,w,c,N,'single');
for n = 1:N
   a = load(XTrain{n},'data');
   X(:,:,:,n) = a.data;
end
X = permute(X,[1 3 2 4]);
end