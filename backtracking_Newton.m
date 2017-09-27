function [gamma,i_k] = backtracking_Newton(alpha,gradient_vector,v_direction,HSIC_target,h_alpha,rho,beta)
%BACKTRACKING_NEWTON Summary of this function goes here
%function to perform backtracking in proximal newton method
%author: Chen Wang, University College London
%Input arguments:
%       alpha: the current value
%       gradient vector: the column vector that stands for the gradient of
%                        g(x)
%       v_direction: the 'proposed' update direction 
%       HSIC_target: handle of the function f(x) = g(x) + h(x)
%       g_alpha: handle of g(x)
%       h_alpha: handle of h(x)
%       rho: the parameter to control the forward step
%       beta: shrinkage constant

%Check if variable gamma is legal input
if rho<0 || rho>0.5
    error('The input rho must be between 0 and 0.5');
end
%Check if variable Eta is legal input
if beta>1
    error('The value of beta should be less than 1');
end
%initialize i_k as the smallest possible non-negative integer
i_k = 0;
gamma = 1;
while(1)
    if i_k>25      %Maximum iteration times
        break;
    end
    step_value = HSIC_target(alpha+gamma*v_direction);
    second_order_value = HSIC_target(alpha) + rho*gamma*(v_direction.')*gradient_vector + rho*(h_alpha(alpha+gamma*v_direction)-h_alpha(alpha));
    if step_value<=second_order_value
        break;
    end
    i_k = i_k + 1;
    gamma = gamma*beta;
end


end

