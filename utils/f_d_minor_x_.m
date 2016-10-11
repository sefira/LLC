function  [numgrad,mangrad] = f_d_minor_x(varargin)
% Compute the number gradient and manual gradient 
% to compare these, if equal, then porve derivation formula is right
    J = @m_function;
    global theta;
    global B;
    global lambda;
    global sigma;
    if nargin >= 4
        theta = varargin{1};
        B = varargin{2};
        lambda = varargin{3};
        sigma = varargin{4};
    else
        theta = [0.2511;
                0.6160;
                0.4733;
                0.3517];
          B = rand(4,10);
%         B = [1,0,0,0,1,0,0,1,0,1;
%             0,1,0,0,1,1,0,1,1,1;
%             0,0,1,0,0,1,1,1,1,1;
%             0,0,0,1,0,0,1,0,1,1];
%         B = [1,0,0,0,;
%             0,1,0,0,;
%             0,0,1,0,;
%             0,0,0,1];
%         B = [0.0,0.2511,0.2511,0.2511;
%             0.6160,0.0,0.6160,0.6160;
%             0.4733,0.4733,0.0,0.4733;
%             0.3517,0.3517,0.3517,0.0];
        lambda = 500;
        sigma = 10;
    end
    % Initialize numgrad with zeros
    size_of_theta = size(theta);
    numgrad = zeros(size_of_theta(1),1);

    EPSILON=1e-6;
    for i=1:length(theta)
        d=zeros(size(theta));
        d(i) = EPSILON;
        numgrad(i)=(J(theta+d/2,B,lambda,sigma)-J(theta-d/2,B,lambda,sigma))/EPSILON;
    end
    
    % show the manual gradient, same as hand write
    mangrad = manual_gradient(theta,B,lambda,sigma);
end

function res = m_function(theta,B,lambda,sigma)
% Origin mapping function
    % x
    x = theta;
    % d
    temp1 = x*ones(length(B),1)'-B;
    temp2 = temp1'*temp1;
    temp2 = temp2/sigma;
    d = diag(temp2);
    % D
    D = diag(d);
    % M
    M = (B'-ones(length(B),1)*x')*(B'-ones(length(B),1)*x')';
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

function mangrad = manual_gradient(theta,B,lambda,sigma)
% Initialize numgrad with zeros
    % x
    x = theta;
    % d
    temp1 = x*ones(length(B),1)'-B;
    temp2 = temp1'*temp1;
    d = diag(temp2);
    % M
    M = (B'-ones(length(B),1)*x')*(B'-ones(length(B),1)*x')';
    [~,mangrad_f_d_minor] = f_d_minor(d,M,lambda);
    
    res = zeros(size(x,1),1);
    for i = 1:size(x,1)
        for j = 1:size(B,2)
            res(i) = res(i) + mangrad_f_d_minor(j) * (2/sigma) * (x(i) - B(i,j));
        end
    end
    mangrad = res;
end

