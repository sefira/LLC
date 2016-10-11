function  [numgrad,mangrad] = f_c_tilde(varargin)
% Compute the number gradient and manual gradient 
% to compare these, if equal, then porve derivation formula is right
    J = @m_function;
    global theta;
    if nargin > 0
        theta = varargin{1};
    else
        theta = [0.8147;0.9058;0.1270;0.9134];
    end
    % Initialize numgrad with zeros
    size_of_theta = size(theta);
    numgrad = zeros(size_of_theta(1),1);

    EPSILON=1e-6;
    for i=1:length(theta)
        d=zeros(size(theta));
        d(i) = EPSILON;
        numgrad(i)=(J(theta+d/2)-J(theta-d/2))/EPSILON;
    end
    
    % show the manual gradient, same as hand write
    mangrad = manual_gradient(theta);
end

function res = m_function(theta)
% Origin mapping function
    % c_tilde
    c_tilde = theta;
    size_of_c_tilde = size(c_tilde);
    % c
    c = c_tilde / (ones(size_of_c_tilde(1),1)'*c_tilde);
    % f:loss: mean square loss, suppose target equal zeros
    f = (1/2)*(sum(c.^2));
    res = f;
end

function mangrad = manual_gradient_f_c(theta)
    size_of_theta = size(theta);
    mangrad = theta / (ones(size_of_theta(1),1)'*theta);
end

function mangrad = manual_gradient(theta)
% Initialize numgrad with zeros
    % compute the gradient of c_c_tilde and f_c
    [~,mangrad_c_c_tilde] = c_c_tilde(theta);
    mangrad_f_c = manual_gradient_f_c(theta);
    
    size_of_theta = size(theta);
    mangrad = zeros(size_of_theta(1),1);
    size_of_mangrad_c_c_tilde = size(mangrad_c_c_tilde);
    for i = 1:size_of_mangrad_c_c_tilde(2)
        for j = 1:size_of_mangrad_c_c_tilde(1)
            mangrad(i) = mangrad(i) + mangrad_f_c(j)*mangrad_c_c_tilde(j,i);
        end
    end
end

