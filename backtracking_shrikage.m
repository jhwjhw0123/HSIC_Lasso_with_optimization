function [gamma,i_k] = backtracking_shrikage(HSIC_target,Q_func,alpha_update_input,y,gamma_prev,Eta,lambda)
%backtracking for the ISTA/FISTA algorithm
%find the least integer i_k that could let the equation hold
%Author: Chen Wang, UCL Dept. of Computer Science
%Input arguments:
%       HSIC_target: Handle of the target function F
%       Q_func: Handle of the Q function for comparison
%       alpha_update_input: handle of the alpha update rountine
%       y: y value for us to backtraking
%       gamma_prev: the value of gamma at the previous iteration
%       lambda: the nomrlization factor used in updating alpha
%       Eta: A number that is greater than 1
%Refrence: Codes in the lecture for gradient decent backtracking

%Check if variable gamma is legal input
if gamma_prev<0
    error('The input gamma must be a number that is greater than 0');
end
%Check if variable Eta is legal input
if Eta<1
    error('The value of Eta should be grater than 1');
end
%initialize i_k as the smallest possible non-negative integer
i_k = 0;
while(1)
    if i_k>25      %Maximum iteration times
        break;
    end
    L_prev = 1/gamma_prev;
    L = L_prev*(Eta^i_k);
    gamma = 1/L;
    alpha = proximal_l1_non_negative(alpha_update_input(y,gamma),gamma*lambda);
    if HSIC_target(alpha)<=Q_func(alpha,y,L)
        break;
    end
    i_k = i_k + 1;
end

end

