function proximal_var = proximal_l1_non_negative(input_var,gamma)
%Input arguments
%input_var: input variable for the non-negative l1 shrinkage
%gamma: the /gamma (or alternatively known as /lambda) variable
%   Detailed explanation goes here
input_var(input_var>=gamma) = input_var(input_var>=gamma)-gamma;
input_var(input_var<gamma) = 0;
proximal_var = input_var;
end

