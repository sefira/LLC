function  [numgrad,mangrad] = c_c_tilde(varargin)
% compute the number gradient and manual gradient 
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
    numgrad = zeros(size_of_theta(1));

    EPSILON=1e-6;
    for i=1:length(theta)
        d=zeros(size(theta));
        d(i) = EPSILON;
        numgrad(:,i)=(J(theta+d/2)-J(theta-d/2))/EPSILON;
    end
    
    % show the manual gradient, same as hand write
    mangrad = manual_gradient(theta);
end

function res = m_function(theta)
% origin mapping function
    size_of_theta = size(theta);
    res = theta / (ones(size_of_theta(1),1)'*theta);
end

function mangrad = manual_gradient(theta)
% Initialize numgrad with zeros
    size_of_theta = size(theta);
    mangrad = zeros(size_of_theta(1));
    denominator = (ones(size_of_theta(1),1)'*theta);
    size_of_numgrad = size(mangrad);
    for i = 1:size_of_numgrad(1)
        for j = 1:size_of_numgrad(2)
            if i == j 
                numerator = denominator - theta(i);
            else
                numerator = -theta(i);
            end
            mangrad(i,j) = numerator / (denominator^2);
        end
    end
end

