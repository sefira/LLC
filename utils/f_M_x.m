function  [numgrad,mangrad] = f_M_x(varargin)
% Compute the number gradient and manual gradient 
% to compare these, if equal, then porve derivation formula is right
    J = @m_function;
    global theta;
    global D;
    global lambda;
    global B;
    if nargin >= 4
        theta = varargin{1};
        D = varargin{2};
        lambda = varargin{3};
        B = varargin{4};
    else
        theta = [0.8147;0.9058;0.1270;2];
        D = [0.2511, 0, 0, 0;
            0, 0.6160, 0, 0;
            0, 0, 0.4733, 0;
            0, 0, 0, 0.3517];
        lambda = 100;
        B = [0.6555,0.2769,0.6948,0.4387;
            0.1712,0.0462,0.3171,0.3816;
            0.7060,0.0971,0.9502,0.7655;
            0.0318,0.8235,0.0344,0.7952];
    end
    % Initialize numgrad with zeros
    size_of_theta = size(theta);
    numgrad = zeros(size_of_theta(1),1);

    EPSILON=1e-6;
    for i=1:length(theta)
        d=zeros(size(theta));
        d(i) = EPSILON;
        numgrad(i)=(J(theta+d/2,D,lambda,B)-J(theta-d/2,D,lambda,B))/EPSILON;
    end
    
    % show the manual gradient, same as hand write
    mangrad = manual_gradient(theta,D,lambda,B);
end

function res = m_function(theta,D,lambda,B)
% Origin mapping function
    % x
    x = theta;
    % M
    M = (B'-ones(length(B),1)*x')*(B'-ones(length(B),1)*x')';
    % c_tilde
    c_tilde = (M + lambda*(D^2));
    size_of_c_tilde = size(c_tilde);
    c_tilde = c_tilde \ ones(size_of_c_tilde(1),1);
    % c
    c = c_tilde / (ones(size_of_c_tilde(1),1)'*c_tilde);
    % f:loss: mean square loss, suppose target equal zeros
    f = (1/2)*(sum(c.^2));
    res = f;
end

function mangrad = manual_gradient(theta,D,lambda,B)
% Initialize numgrad with zeros
    % x
    x = theta;
    % B
    B = (B'-ones(length(B),1)*x');
    % M 
    M = B * B';
    % f_M
    [~,mangrad_f_M] = f_M(M,D,lambda);
    res1 = ones(length(B),1)' * mangrad_f_M * B;
    res2 = ones(length(B),1)' * mangrad_f_M' * B;
    mangrad = -(res1+res2)';
end

