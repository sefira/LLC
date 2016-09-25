local LLC, parent = torch.class('nn.LLC', 'nn.Module')

function LLC:__init(B, lambda, sigma)
   parent.__init(self)
   -- dictionary
   -- print(B)
   if B == nil then
      self.B = torch.Tensor(
            {{0.0,0.2511,0.2511,0.2511,-10},
            {0.6160,0.0,0.6160,0.6160,-10},
            {0.4733,0.4733,0.0,0.4733,-10},
            {0.3517,0.3517,0.3517,0.0,-10}})
      self.B = torch.rand(5,10)
      -- print "B"
      -- print(self.B)
   else
      self.B = B
   end
   
   -- coefficient of D
   if lambda == nil then
      self.lambda = 500
   else
      self.lambda = lambda
   end
   
   -- coefficient of exp(d/sigma)
   if sigma == nil then
      self.sigma = 100
   else
      self.sigma = sigma
   end
   self.d = 0
   self.M = 0
   
end

-- encode to LLC
function LLC:updateOutput(input)
   local x = input:resize((#input)[1],1)
   local B = self.B
   local sigma = self.sigma
   local lambda = self.lambda
   local ones_B = torch.ones((#B)[2],1)
   -- d
   local temp1 = x*ones_B:t()-B
   local temp2 = temp1:t()*temp1
   self.d = torch.diag(temp2)
   self.d = self.d / (sigma)
   -- D
   local D = torch.diag(self.d)
   -- M
   self.M = (B:t()-ones_B*x:t())*(B:t()-ones_B*x:t()):t()
   -- c_tilde
   local c_tilde = (self.M+lambda*(torch.mm(D,D)))
   c_tilde = torch.inverse(c_tilde) * torch.ones((#c_tilde)[1],1)
   -- c
   local ones_c_tilde = torch.ones((#c_tilde)[1],1)
   local denominator = torch.repeatTensor((ones_c_tilde:t()*c_tilde),#c_tilde)
   local c = torch.cdiv(c_tilde,denominator)
   self.output = c:reshape((#c)[1])
   return self.output
end

function LLC:updateGradInput(input, gradOutput)
   self.gradOutput = gradOutput
   local x = input:resize((#input)[1],1)
   local B = self.B
   local sigma = self.sigma
   local lambda = self.lambda
   local ones_B = torch.ones((#B)[2],1)
   local M = self.M
   local d = self.d
   -- D
   local D = torch.diag(d)
   local mangrad_f_d_minor_x = self:f_d_minor_x(x,d,B,lambda,sigma,M)
   local mangrad_f_M_x = self:f_M_x(x,D,lambda,B,M)
   local mangrad = mangrad_f_d_minor_x + mangrad_f_M_x
   self.gradInput = mangrad:reshape((#mangrad)[1])
   return self.gradInput
end

function LLC:f_d_minor_x(x,d,B,lambda,sigma,M)
    local ones_B = torch.ones((#B)[2],1)
    -- f_d_minor
    local mangrad_f_d_minor = self:f_d_minor(d,M,lambda)
    -- B_tilde2 is the inner product of (x1:t() - B)
    local mangrad_f_B_tilde2 = torch.diag(mangrad_f_d_minor)
    mangrad_f_B_tilde2 = mangrad_f_B_tilde2/sigma
    
    local mangrad_f_B_tilde = 2*(x*ones_B:t()-B)*mangrad_f_B_tilde2
    local mangrad_f_x = mangrad_f_B_tilde * ones_B
    local mangrad = mangrad_f_x
    return mangrad
end

function LLC:f_d_minor(d,M,lambda)
    local D = torch.diag(d)
    local mangrad_f_D = self:f_D(D,M,lambda)
    local mangrad = torch.diag(mangrad_f_D)
    return mangrad
end

function LLC:f_D(D,M,lambda)
    -- f_c_tilde
    local D_tilde = (M + lambda*(torch.mm(D,D)))
    local A = -lambda * (torch.inverse(D_tilde))
    local c_tilde = torch.inverse(D_tilde) * torch.ones((#D_tilde)[1],1)
    local B = c_tilde
    local mangrad_f_c_tilde = self:f_c_tilde(c_tilde)
    local mangrad = (D*B*mangrad_f_c_tilde:t()*A + B*mangrad_f_c_tilde:t()*A*D):t()
    return mangrad
end

function LLC:f_M_x(x,D,lambda,B,M)
    -- B_tilde
    local ones_B = torch.ones((#B)[2],1)
    local B_tilde = (B:t()-ones_B*x:t())
    local ones_B_tilde = torch.ones((#B_tilde)[1],1)
    -- f_M
    local mangrad_f_M = self:f_M(M,D,lambda)
    -- f_M_x
    local res1 = ones_B_tilde:t() * mangrad_f_M * B_tilde
    local res2 = ones_B_tilde:t() * mangrad_f_M:t() * B_tilde
    local mangrad = -(res1+res2):t()
    return mangrad
end

function LLC:f_M(M,D,lambda)
    -- c_tilde
    local M_tilde = (M + lambda*(torch.mm(D,D)))
    local c_tilde = M_tilde
    local ones_c_tilde = torch.ones((#c_tilde)[1],1)
    c_tilde = torch.inverse(c_tilde) * ones_c_tilde
    -- f_c_tilde
    local mangrad_f_c_tilde = self:f_c_tilde(c_tilde)
    -- f_M
    local ones_M_tilde = torch.ones((#M_tilde)[1],1)
    local M_tilde_inverse = torch.inverse(M_tilde) 
    local res = M_tilde_inverse * ones_M_tilde * mangrad_f_c_tilde:t() * M_tilde_inverse
    local mangrad = -res:t()
    return mangrad
end

function LLC:f_c_tilde(c_tilde)
    -- c_c_tilde
    local mangrad_c_c_tilde = self:c_c_tilde(c_tilde)
    -- f_c
    local ones_c_tilde = torch.ones((#c_tilde)[1],1)
    local c = c_tilde / (ones_c_tilde:t()*c_tilde)[1][1]
    local mangrad_f_c = self.gradOutput
    -- f_c_tilde
    local mangrad = torch.zeros((#c_tilde)[1],1)
    for i = 1,(#mangrad_c_c_tilde)[2] do
        for j = 1,(#mangrad_c_c_tilde)[1] do
            mangrad[i] = mangrad[i] + mangrad_f_c[j]*mangrad_c_c_tilde[j][i]
        end
    end
    return mangrad
end

function LLC:c_c_tilde(c_tilde)
    local mangrad = torch.zeros((#c_tilde)[1],(#c_tilde)[1])
    local ones_c_tilde = torch.ones((#c_tilde)[1],1)
    local denominator = ones_c_tilde:t()*c_tilde
    denominator = denominator[1][1]
    for i = 1,(#mangrad)[1] do
        for j = 1,(#mangrad)[2] do
            local numerator = 0
            if i == j then
                numerator = denominator - c_tilde[i]
            else
                numerator = -c_tilde[i]
            end
            mangrad[i][j] = numerator / denominator^2
        end
    end
    return mangrad
end


