function numgrad = computeNumericalGradient(varargin)
% numgrad = computeNumericalGradient(J, theta)
% numgrad = computeNumericalGradient(@a_function,[4,3,2,1])
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the
% function value at theta. 
    J = @m_function;
    theta = [0.8147,0.9058,0.1270,0.9134];
    %theta = zeros(1,4)
    % Initialize numgrad with zeros
    numgrad = zeros(size(theta));

    %% ---------- YOUR CODE HERE --------------------------------------
    % Instructions: 
    % Implement numerical gradient checking, and return the result in numgrad.  
    % (See Section 2.3 of the lecture notes.)
    % You should write code so that numgrad(i) is (the numerical approximation to) the 
    % partial derivative of J with respect to the i-th input argument, evaluated at theta.  
    % I.e., numgrad(i) should be the (approximately) the partial derivative of J with 
    % respect to theta(i).
    %                
    % Hint: You will probably want to compute the elements of numgrad one at a time. 

    % J0 = J(theta, varargin{:});
    EPSILON=1e-6;
    for i=1:length(theta)
        d=zeros(size(theta));
        d(i) = EPSILON;
        numgrad(i)=(J(theta+d/2,varargin{:})-J(theta-d/2,varargin{:}))/EPSILON;
    end

end

function res = m_function(theta)
res = theta(1)^2 + theta(2)^3 + theta(3) + theta(4)^12;
end

