function [alpha,optvalue,info] = ADMM_least_square_lasso(alpha_0,HSIC_target,A,b,lambda,maxInteration,rho,rho_update,rho_max,reltol,abstol,update_mode,stop_criteria)
%ADMM function specified for least square lasso with non-negative
%constraint
%The proximal oeprator could be explicitly solved (analytically solved)
%Thus this rountine is fast
%Author: Chen Wang
%Input args: 
%       alpha_0: initial value of target variable alpha_0
%       HSIC_target: target optimization function
%       A: The A matrix for least square
%       b: The b vector for least square
%       lambda: pernalization factor
%       maxInteration: maximum iteration times
%       rho: The parameter rho in ADMM method
%       rho_update: The parameter controls whether updata rho through time
%       rho_max: The maximum permitted value of rho. Only make difference
%                when rho_update==true.
%       reltol: relative stop tolerance
%       abstol: absolute stop tolerance
%Other variables:
%       z: The variable for splitted f(x)+g(z)
%       u: The dual variable in Augmented Lagrangian
%       update_mode: 
%           'CG': Conjugate Gradient to update alpha
%           'INV': Matrix inverse method to update alpha

%Define a information structual object and return useful informations
info.steps = 1;
info.alpha_value = zeros(size(alpha_0,1),maxInteration);  %Dim * Iteration
info.alpha_value(:,1) = alpha_0;
info.converge = false;
%get the dimension of alpha
nDim = size(alpha_0,1);
mAmount = size(A,1);
%start with gamma_0
gamma = 1;
%initialize alpha
alpha = alpha_0;
%initialize z and u
z = -alpha_0;
%Ax+Bz=c, here A=1, B=-1, c = 0
u = 0;
%Pre-compute matrix to save computational time
AA = A.'*A;
Ab = A.'*b;
if strcmpi(update_mode, 'INV')
    %define the function handle
    alpha_update = @(rho,z_k,u_k,gamma)mldivide((rho*eye(size(AA))+gamma*AA),(rho*z_k-u_k-gamma*Ab));
end
%proximal stopping cretiria function
f_gamma = @(x,y,gamma_k)(((1/2)*(A*y-b).'*(A*y-b))+(AA*y-Ab).'*(x-y)+(1/2*gamma_k)*(x-y).'*(x-y));
%iterately update alpha
for k = 1:1:maxInteration
    alpha_k_1 = alpha;
    %Update alpha
    switch upper(update_mode)
        case 'INV'
            alpha = alpha_update(rho,z,u,gamma);
            alpha(alpha<=0) = 0;
        case 'CG'
            %Conjugate Gradient Routine
            alpha = pcg(AA+rho*eye(size(AA)),rho*(u-alpha));
    end
    %store previous z
    z_k_1 = z;
    %update z
    z = proximal_l1_non_negative(alpha+(u/rho),gamma*lambda/rho);
    %update u (dual variable)
    u = u + rho*(alpha-z);
    %Updata rho (optional)
    %quote from  Section 3.3 in Boyd's book: strategies for adapting rho
    if rho_update==true
        rho = min(rho_max,1.1*rho); 
    end
    %Store the information
    info.alpha_value(:,k) = alpha;
    info.steps = k;
    %Determin to use which kind of stop condition
    switch lower(stop_criteria)
        case 'norm'
            stop_condition = (norm(alpha-alpha_k_1)<=reltol);
        case 'residual'
            %compute residual to judge when to stop
            %Reference: This part (residual criteria) is a revision version combining 
            %   http://web.stanford.edu/~boyd/papers/admm/lasso/lasso.html
            %   and
            %   The codes in the lecture
            r_norm  = norm(alpha - z);
            s_norm  = norm(-rho*(z - z_k_1));
            eps_pri = sqrt(nDim)*abstol + reltol*max(norm(alpha), norm(-z));
            eps_dual = sqrt(mAmount)*abstol + reltol*norm(rho*u);
            stop_condition = (max(r_norm/eps_pri,s_norm/eps_dual)<1);
    end
    if stop_condition == 1
        info.converge = true;
        break
    end
end

optvalue = HSIC_target(alpha);

end

