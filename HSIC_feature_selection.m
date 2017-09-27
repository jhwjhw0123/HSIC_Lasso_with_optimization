function [alpha,optvalue,info] = HSIC_feature_selection(Data_X,Data_Y,lambda,mode)
%The main function to implement HSIC Lasso
%Here we will try to implement different optimisation methods 
%Calculate the parameters
nData = size(Data_X,1);
nDim = size(Data_X,2);
if nData ~= size(Data_Y,1)
    error('The input data amounts of X and Y must be the same!');
end
sigma = 0.01;     %Hyper-parameter for the Gaussian kernel
Gamma = eye(nData) - ones(nData,nData)./nData;
L = Gaussian_Kernel(Data_Y,Data_Y,sigma);
L_Gamma = Gamma*L*Gamma;
%This problem could be vectorized and tranfered into a non-negative
%quodratic lasso problem
%A and b for vectorized computation
b = L_Gamma(:);
A = zeros(nData*nData,nDim);
K_Gamma = zeros(nData,nData);
for cDim = 1:1:nDim
    K_k = Gaussian_Kernel(Data_X(:,cDim),Data_X(:,cDim),sigma);
    K_k_Gamma = Gamma*K_k*Gamma;
    K_Gamma = K_Gamma + K_k_Gamma;
    A(:,cDim) = K_k_Gamma(:);
end
%% Solve the problem
HSIC_target = @(alpha_v)(1/2)*(A*alpha_v-b).'*(A*alpha_v-b)+lambda*norm(alpha_v,1);
alpha_0 = ones(nDim,1);      %initial value, this will be kept consistent for comparison
maxInteration = 500;
tol = 1e-4; 
tic;
switch mode
    case 'proximal_gradient'
%% Proximal Gradient (implement)
    %Avaliable stopping cretiria: 'norm' 'fz'
    [alpha,optvalue,info] = proximal_gradient(alpha_0,HSIC_target,A,b,lambda,maxInteration,tol,'fz');
    case 'acc_proximal_gradient'
%% Accelerated Proximal Gradient (implement)
    %Avaliable stopping cretiria: 'norm' 'fz'
    [alpha,optvalue,info] = accelerated_proximal_gradient(alpha_0,HSIC_target,A,b,lambda,maxInteration,tol,'fz');
%% FISTA (implement)
    case 'FISTA_const'
    [alpha,optvalue,info] = FISTA_constraint(alpha_0,HSIC_target,A,b,lambda,maxInteration,tol,'fz','constant');
    case 'FISTA_backtrack'
    [alpha,optvalue,info] = FISTA_constraint(alpha_0,HSIC_target,A,b,lambda,maxInteration,tol,'fz','backtrack');
%% Projected Newton (implement)
    case 'Newton_proximal_const'
    epsilon = 5*1e-2;
    [alpha,optvalue,info] = projected_newton_constraint(alpha_0,HSIC_target,A,b,lambda,maxInteration,tol,epsilon,'norm','const');
    case 'Newton_proximal_backtrack'
    epsilon = 5*1e-2;
    [alpha,optvalue,info] = projected_newton_constraint(alpha_0,HSIC_target,A,b,lambda,maxInteration,tol,epsilon,'norm','backtrack');
%% ADMM Method (implement)
    case 'ADMM'
    rho = 1;
    reltol = 1e-4;            %tolerance for stop optimization
    abstol = 1e-4;
    [alpha,optvalue,info] = ADMM_least_square_lasso(alpha_0,HSIC_target,A,b,lambda,maxInteration,rho,true,3,reltol,abstol,'INV','residual');
%% DAL(implement)
    case 'DAL'
    [alpha,optvalue,info] = Dual_augmented_lagrange_least_square(alpha_0,HSIC_target,A,b,lambda,maxInteration,tol,'fz','inv');
%% DAL method (package)
    case 'DAL_package'
    [alpha,stat]=dalsql1(ones(nDim,1), A, b, lambda);
%% Quasi-Newton proximal splitting (package)
% V_dim = size(A,2);
% prox= @(alpha_0,d,u,varargin) prox_rank1_l1pos( alpha_0, d, u, lambda, [], varargin{:} );
% h = @(alpha_v) lambda*norm(alpha_v,1);
% fcnGrad = @(alpha_v) normSquaredFunction(alpha_v,A,[],b);
% opts = struct('N',V_dim,'verbose',0,'nmax',4000,'tol',1e-13);
% %opts.L = normQ;    % optional
% [alpha,nit, errStruct,optsOut] = zeroSR1(fcnGrad,[],h,prox,opts);
end
toc;
t = toc;
info.time = t;
alpha = alpha./max(alpha);
optvalue = HSIC_target(alpha);
end

