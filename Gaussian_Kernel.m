function Kernel_output = Gaussian_Kernel(DataLeft,DataRight,sigma)
% This function produce a Gaussian kernel for the HSIC Lasso.
% This function uses vectorized method so that the computational time is
% saved
% Author: Chen Wang, UCL Dept. of Computer Science
%   Detailed explanation goes here
nDataLeft = size(DataLeft,1);
nDim = size(DataLeft,2);
nDataRight = size(DataRight,1);
if nDim ~= size(DataRight,2)
    error('The dimensions of two input data must be the same!');
end
kernel_left = reshape(repmat(DataLeft,1,nDataRight).',nDim,nDataLeft*nDataRight).';
kernel_right = repmat(DataRight,nDataLeft,1);
Kernel_output = reshape(exp(-sum((kernel_left-kernel_right).^2,2)/2*sigma^2),nDataRight,nDataLeft).';
%The above rountine could gaurantee the output kernel be M*N, where M is
%the amount of the left input and N is the amount of the rifht input.
%Athough in this problem it is won't make a obvious difference
end

