function I = readMatFile(filename)
I = load(filename);
I = I.data;
I = permute(I,[1,3,2]);
I = I(:,:,1);
end