function [alpha,optvalue,info] = accelerated_proximal_gradient(alpha_0,HSIC_target,A,b,lambda,maxInteration,tol,stop_criteria);
%Function for least square with non-negative lasso 
%with accelerated proximal fradient method
%This is similar to the oringinal proximal_gradient version
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

%Define a information structual object and return useful informations
info.steps = 1;
info.alpha_value = zeros(size(alpha_0,1),maxInteration);  %Dim * Iteration
info.alpha_value(:,1) = alpha_0;
info.converge = false;
%start with gamma_0
gamma_0 = 1;
%Discounted with 0.5
beta = 0.5;
%Pre-compute matrix to save computational time
AA = A.'*A;
Ab = A.'*b;
%define the function handle
%non-negative lasso input arguments
alpha_update_input = @(y_k,gamma_k) (y_k-gamma_k*AA*y_k+gamma_k*Ab);
y_update = @(alpha_k_1,alpha_k_2,w_k)(alpha_k_1 + w_k*(alpha_k_1-alpha_k_2));
%proximal stopping cretiria function
f_gamma = @(x,y,gamma_k)(((1/2)*(A*y-b).'*(A*y-b))+(AA*y-Ab).'*(x-y)+(1/2*gamma_k)*(x-y).'*(x-y));
%compute alpha at the fisrt interation
alpha = proximal_l1_non_negative(alpha_update_input(alpha_0,gamma_0),gamma_0*lambda);
gamma = gamma_0 * beta;
alpha_k_2 = alpha_0;
%iterately update alpha
for k = 2:1:maxInteration
    alpha_k_1 = alpha;
    %compute w_k
    w_k = k/(k+3);
    %update y
    y_current = y_update(alpha_k_1,alpha_k_2,w_k);
    alpha = proximal_l1_non_negative(alpha_update_input(y_current,gamma),gamma*lambda);
    %update alpha_{k-2}
    alpha_k_2 = alpha_k_1;
    %Updata gamma
    gamma = gamma * beta;
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

