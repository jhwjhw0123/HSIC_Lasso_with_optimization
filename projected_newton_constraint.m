function [alpha,optvalue,info] = projected_newton_constraint(alpha_0,HSIC_target,A,b,lambda,maxInteration,tol,epsilon,stop_criteria,step_mode)
%Function for least square with non-negative lasso 
%with proximal Newton method
%Author: Chen Wang, UCL Dpet. of Computer Science
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
%       epsilon: The value for hessian to stop work 
%       step_type:
%           'const': constant step size with iterative constatly shriking strategy
%           'backtrack': using backtracking method to determine the step
%           size

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
%Handle of gradient
gradient_handle = @(alpha_k)(A.'*(A*alpha_k-b));
%Handle of inverse Hessian 
%Add 0.01 diagonal to ensure stability
Hessian_inverse = @(A)(inv(2*A.'*A+0.01*eye(size(A,2))));
%non-negative lasso input arguments
alpha_update_gradient = @(alpha_k,gamma_k,gradient) (alpha_k-gamma_k*gradient);
alpha_update_hessian = @(alpha_k,gamma_k,gradient,Hessian_inv)(alpha_k-gamma_k*Hessian_inv*gradient);
%proximal stopping cretiria function
f_gamma = @(x,y,gamma_k)(((1/2)*(A*y-b).'*(A*y-b))+(AA*y-Ab).'*(x-y)+(1/2*gamma_k)*(x-y).'*(x-y));
%If using backtracking we need to define g(x) and h(x) seperately
if strcmpi(step_mode, 'backtrack')
    rho = 0.25;
    l1_reg_obg = @(alpha_v)(lambda*norm(alpha_v,1));
end
%Assign initial values
alpha = alpha_0;
gamma = gamma_0;
%iterately update alpha
for k = 1:1:maxInteration
    alpha_k_1 = alpha;
    %Find the gradient-update variables
    close_zeros_index = find(alpha<=epsilon);
    gradient_vector = gradient_handle(alpha);
    positive_grad_index = find(gradient_vector>0);
    gradient_update_ind = intersect(close_zeros_index,positive_grad_index);
    %update the alpha values based on gradient
    aplha_gridient_updated = proximal_l1_non_negative(alpha_update_gradient(alpha(gradient_update_ind),gamma,gradient_vector(gradient_update_ind)),gamma*lambda);
    alpha(gradient_update_ind) = aplha_gridient_updated;
    %Find the Hessian -update variables
    full_ind = 1:1:size(A,2);
    Hessian_update_ind = setdiff(full_ind,gradient_update_ind);
    A_hessian_used = A(:,Hessian_update_ind);
    Hessian_inv = Hessian_inverse(A_hessian_used);
    if strcmpi(step_mode, 'backtrack')
        %line search to update gamma
        v_direction = proximal_l1_Hessian_non_negative(alpha_update_hessian(alpha,gamma,gradient_vector,Hessian_inverse(A)),gamma*lambda)-alpha;
        [gamma,i_k] = backtracking_Newton(alpha,gradient_vector,v_direction,HSIC_target,l1_reg_obg,rho,beta);
    end
    alpha_Hessian_updated = proximal_l1_Hessian_non_negative(alpha_update_hessian(alpha(Hessian_update_ind),gamma,gradient_vector(Hessian_update_ind),Hessian_inv),gamma*lambda);
    alpha(Hessian_update_ind) = alpha_Hessian_updated;
    if strcmpi(step_mode, 'const')
        %Updata gamma
        gamma = gamma * beta;
    end
    %Store the current value
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


