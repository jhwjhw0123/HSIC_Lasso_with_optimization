function [alpha,optvalue,info] = FISTA_constraint(alpha_0,HSIC_target,A,b,lambda,maxInteration,tol,stop_criteria,step_mode)
%Function for FISTA method with non-negative proximal
%This is based on the 2009 literature 'A Fast Iterative Shrinkage-Thresholding Algorithm
%for Linear Inverse Problems' by A. Beck and M. Teboulle
%The basic idea is to compute the proximal based not only on the previous
%step, but also on more steps
%Author: Chen Wang
%Input args: 
%       alpha_0: initial value of target variable alpha_0
%       HSIC_target: target optimization function
%       A: The A matrix for least square
%       b: The b vector for least square
%       lambda: pernalization factor
%       maxInteration: maximum iteration times
%       tol: tolerance for stopping
%       stop_criteria: 
%           'norm': Using the difference between the norm of two iteration
%           to determing the stop criteria
%           'fz': Using proximal judging criteria f(z)<f(z,x)
%       step_mode:
%           'constant': Use constant step size
%           'backtrack': Use backtracking step size

%Define a information structual object and return useful informations
info.steps = 1;
info.alpha_value = zeros(size(alpha_0,1),maxInteration);  %Dim * Iteration
info.alpha_value(:,1) = alpha_0;
info.converge = false;
%start with gamma_0
gamma_0 = 1;
%define t_0 = 1
t_0 = 1;
%Discounted with 0.5
beta = 0.5;
%Pre-compute matrix to save computational time
AA = A.'*A;
Ab = A.'*b;
%define the function handle
%If backtracking, define Q function for computation
if strcmpi(step_mode, 'backtrack')
    % Difine the hyper-parameter eta
    Eta = 1.1;
    Q_func = @(x,y,L)(((1/2)*(A*y-b).'*(A*y-b))+(AA*y-Ab).'*(x-y)+(L/2)*(x-y).'*(x-y)+norm(x,1));
end
%non-negative lasso input arguments
alpha_update_input = @(alpha_k,gamma_k) (alpha_k-gamma_k*AA*alpha_k+gamma_k*Ab);
%t_update formula
t_update = @(t_k)(1+sqrt(1+4*t_k^2)/2);
%y_update formula
y_update = @(alpha_k,alpha_k_1,t_k,t_k_1)(alpha_k + (t_k_1-1/t_k) *(alpha_k-alpha_k_1));
%proximal stopping cretiria function
f_gamma = @(x,y,gamma_k)(((1/2)*(A*y-b).'*(A*y-b))+(AA*y-Ab).'*(x-y)+(1/2*gamma_k)*(x-y).'*(x-y));
%compute alpha at the fisrt interation
y = alpha_0;
switch lower(step_mode)
    case 'constant'
        gamma = gamma_0 * beta;
    case 'backtrack'
        [gamma,i_k] = backtracking_shrikage(HSIC_target,Q_func,alpha_update_input,y,gamma_0,Eta,lambda); 
end
alpha = proximal_l1_non_negative(alpha_update_input(y,gamma_0),gamma_0*lambda);
t = t_update(t_0);
y = y_update(alpha,alpha_0,t,t_0);
%iterately update alpha
for k = 2:1:maxInteration
    %Updata gamma
    switch lower(step_mode)
        case 'constant'
            gamma = gamma * beta;
        case 'backtrack'
            [gamma,i_k] = backtracking_shrikage(HSIC_target,Q_func,alpha_update_input,y,gamma,Eta,lambda);
    end
    alpha_k_1 = alpha;
    %update alpha
    alpha = proximal_l1_non_negative(alpha_update_input(y,gamma),gamma*lambda);
    %update t
    t_prev = t;
    t = t_update(t);
    %update y
    y = y_update(alpha,alpha_k_1,t,t_prev);
    info.alpha_value(:,k) = alpha;
    info.steps = k;
    %Determin to use which kind of stop condition
    switch lower(stop_criteria)
        case 'norm'
            stop_condition = (norm(alpha-alpha_k_1)<=tol);
        case 'fz'
            stop_condition = ((1/2)*(A*alpha-b).'*(A*alpha-b)<=f_gamma(alpha,alpha_k_1,gamma));
    end
    if stop_condition == 1
        info.converge = true;
        break
    end
end

optvalue = HSIC_target(alpha);

end

