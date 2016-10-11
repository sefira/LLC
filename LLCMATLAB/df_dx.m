function [code,mangrad,numgrad] = df_dx(varargin)
% Output the final encode result
% Compute the number gradient and manual gradient 
% to compare these, if equal, then porve derivation formula is right
    global x;
    global B;
    global lambda;
    global sigma;
    if nargin >= 4
        x = varargin{1};
        B = varargin{2};
        lambda = varargin{3};
        sigma = varargin{4};
    else
        x = rand(4,1);
        B = rand(4,10);
        x = [0.2511;
            0.6160;
            0.4733;
            0.3517];
        B = [0.0,0.2511,0.2511,0.2511,-10;
            0.6160,0.0,0.6160,0.6160,-10;
            0.4733,0.4733,0.0,0.4733,-10;
            0.3517,0.3517,0.3517,0.0,-10];
        lambda = 500;
        sigma = 100;
    end
    % code
    [code,~] = forward(x,B,lambda,sigma);
    
    % number gradient
    n = size(x,1);
    numgrad = zeros(n,1);

    EPSILON=1e-6;
    for i=1:n
        d=zeros(n,1);
        d(i) = EPSILON;
        [~,a] = forward(x+d/2,B,lambda,sigma);
        [~,b] = forward(x-d/2,B,lambda,sigma);
        numgrad(i) = (a-b)/EPSILON;
    end
    
    % manual gradient, computed by derivation
    mangrad = backward(x,B,lambda,sigma);
end

function [c,res] = forward(theta,B,lambda,sigma)
% Origin mapping function
    % x
    x = theta;
    % d
    temp1 = x*ones(size(B,2),1)'-B;
    temp2 = temp1'*temp1;
    d = diag(temp2);
    d = d / sigma;
    % D
    D = diag(d);
    % M
    M = (B'-ones(size(B,2),1)*x')*(B'-ones(size(B,2),1)*x')';
    % c_tilde
    c_tilde = (M+lambda*(D^2));
    c_tilde = c_tilde \ ones(size(c_tilde,1),1);
    % c
    c = c_tilde / (ones(size(c_tilde,1),1)'*c_tilde);
    % f:loss: mean square loss, suppose target equal zeros
    loss = lossfunction(c);
    res = loss;
end

function loss = lossfunction(c)
    loss = (1/2)*(sum(c.^2));
end

function mangrad = backward(theta,B,lambda,sigma)
% Initialize numgrad with zeros
    % x
    x = theta;
    % d
    temp1 = x*ones(size(B,2),1)'-B;
    temp2 = temp1'*temp1;
    d = diag(temp2);
    d = d/sigma;
    % D
    D = diag(d);
    % M
    M = (B'-ones(size(B,2),1)*x')*(B'-ones(size(B,2),1)*x')';
    
    mangrad_f_d_minor_x = f_d_minor_x(x,d,B,lambda,sigma,M);
    mangrad_f_M_x = f_M_x(x,D,lambda,B,M);
    mangrad = mangrad_f_d_minor_x + mangrad_f_M_x;
end

function mangrad = f_d_minor_x(x,d,B,lambda,sigma,M)
    % f_d_minor
    mangrad_f_d_minor = f_d_minor(d,M,lambda);
    % B_tilde2 is the inner product of (x1' - B)
    mangrad_f_B_tilde2 = diag(mangrad_f_d_minor);
    mangrad_f_B_tilde2 = mangrad_f_B_tilde2/sigma;
    
    mangrad_f_B_tilde = 2*(x*ones(size(B,2),1)'-B)*mangrad_f_B_tilde2;
    mangrad_f_x = mangrad_f_B_tilde * ones(size(B,2),1);
    mangrad = mangrad_f_x;
end

function mangrad = f_d_minor(d,M,lambda)
    D = diag(d);
    mangrad_f_D = f_D(D,M,lambda);
    mangrad = diag(mangrad_f_D);
end

function mangrad = f_D(D,M,lambda)
    % f_c_tilde
    D_tilde = (M + lambda*(D^2));
    A = -lambda * (inv(D_tilde));
    c_tilde = D_tilde \ ones(size(D_tilde,1),1);
    B = c_tilde;
    mangrad_f_c_tilde = f_c_tilde(c_tilde);
    mangrad = (D*B*mangrad_f_c_tilde'*A + B*mangrad_f_c_tilde'*A*D)';
end

function mangrad = f_M_x(x,D,lambda,B,M)
    % B_tilde
    B_tilde = (B'-ones(size(B,2),1)*x');
    % f_M
    mangrad_f_M = f_M(M,D,lambda);
    % f_M_x
    res1 = ones(size(B_tilde,1),1)' * mangrad_f_M * B_tilde;
    res2 = ones(size(B_tilde,1),1)' * mangrad_f_M' * B_tilde;
    mangrad = -(res1+res2)';
end

function mangrad = f_M(M_tilde,D,lambda)
    % c_tilde
    M_tilde = (M_tilde + lambda*(D^2));
    c_tilde = M_tilde;
    c_tilde = c_tilde \ ones(size(c_tilde,1),1);
    % f_c_tilde
    mangrad_f_c_tilde = f_c_tilde(c_tilde);
    % f_M
    res = M_tilde\ones(size(M_tilde,1),1)*mangrad_f_c_tilde'/M_tilde;
    mangrad = -res';
end

function mangrad = f_c_tilde(c_tilde)
    % c_c_tilde
    mangrad_c_c_tilde = c_c_tilde(c_tilde);
    % f_c
    c = c_tilde / (ones(size(c_tilde,1),1)'*c_tilde);
    mangrad_f_c = f_c(c);
    % f_c_tilde
    mangrad = zeros(size(c_tilde,1),1);
    for i = 1:size(mangrad_c_c_tilde,2)
        for j = 1:size(mangrad_c_c_tilde,1)
            mangrad(i) = mangrad(i) + mangrad_f_c(j)*mangrad_c_c_tilde(j,i);
        end
    end
end

function mangrad = c_c_tilde(c_tilde)
    mangrad = zeros(size(c_tilde,1));
    denominator = (ones(size(c_tilde,1),1)'*c_tilde);
    for i = 1:size(mangrad,1)
        for j = 1:size(mangrad,2)
            if i == j 
                numerator = denominator - c_tilde(i);
            else
                numerator = -c_tilde(i);
            end
            mangrad(i,j) = numerator / (denominator^2);
        end
    end
end

% the loss function
function mangrad = f_c(c)
    res = c;
    mangrad = res;
end



