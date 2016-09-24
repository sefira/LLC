function  [numgrad,mangrad] = f_D(varargin)
% Compute the number gradient and manual gradient 
% to compare these, if equal, then porve derivation formula is right
    J = @m_function;
    global theta;
    global M;
    global lambda;
    if nargin >= 3
        theta = varargin{1};
        M = varargin{2};
        lambda = varargin{3};
    else
        theta = [0.2511, 0, 0, 0;
                0, 0.6160, 0, 0;
                0, 0, 0.4733, 0;
                0, 0, 0, 0.3517];
        M = [0.4898,0.7547,0.1626,0.3404;
            0.4456,0.2760,0.1190,0.5853;
            0.6463,0.6797,0.4984,0.2238;
            0.7094,0.6551,0.9597,0.7513];
        lambda = 100;
    end
    % Initialize numgrad with zeros
    size_of_theta = size(theta);
    numgrad = zeros(size_of_theta(1));

    EPSILON=1e-6;
    for i=1:length(theta)
        for j = 1:length(theta)
            d=zeros(size(theta));
            d(i,j) = EPSILON;
            numgrad(i,j)=(J(theta+d/2,M,lambda)-J(theta-d/2,M,lambda))/EPSILON;
        end
    end
    
    % show the manual gradient, same as hand write
    mangrad = manual_gradient(theta,M,lambda);
end

function res = m_function(theta,M,lambda)
% Origin mapping function
    % D
    D = theta;
    % c_tilde
    c_tilde = (M+lambda*(D^2));
    size_of_c_tilde = size(c_tilde);
    c_tilde = c_tilde \ ones(size_of_c_tilde(1),1);
    % c
    c = c_tilde / (ones(size_of_c_tilde(1),1)'*c_tilde);
    % f:loss: mean square loss, suppose target equal zeros
    f = (1/2)*(sum(c.^2));
    res = f;
end

function mangrad = manual_gradient(theta,M,lambda)
% Initialize numgrad with zeros
    D = theta;
    % compute the gradient of f_c_tilde
    c_tilde = (M + lambda*(D^2));
    A = -lambda * (inv(c_tilde));
    size_of_c_tilde = size(c_tilde);
    c_tilde = c_tilde \ ones(size_of_c_tilde(1),1);
    B = c_tilde;
    [~,mangrad_f_c_tilde] = f_c_tilde(c_tilde);
    mangrad = (D*B*mangrad_f_c_tilde'*A + B*mangrad_f_c_tilde'*A*D)';
end

