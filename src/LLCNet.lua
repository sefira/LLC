local LLCNet, parent = torch.class('nn.LLCNet', 'nn.Module')

function LLCNet:__init(nInputPlane, nOutputPlane, B, lambda, sigma)
   parent.__init(self)
   -- dictionary
   -- print(B)
   if B == nil then
      self.B = torch.Tensor(
            {{0.0,0.2511,0.2511,0.2511,-10},
            {0.6160,0.0,0.6160,0.6160,-10},
            {0.4733,0.4733,0.0,0.4733,-10},
            {0.3517,0.3517,0.3517,0.0,-10}})
      self.B = torch.rand(nInputPlane,nOutputPlane)
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

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.d = 0
   self.M = 0
   self.column_row = 0
   self.column_column = 0
end

-- encode to LLCNet
function LLCNet:updateOutput(input)
   -- init parameters   
   local B = self.B
   local sigma = self.sigma
   local lambda = self.lambda
   local ones_B = torch.ones(self.nOutputPlane,1)
   self.d = torch.zeros(input:size(2),input:size(3),self.nOutputPlane)
   self.M = torch.zeros(input:size(2),input:size(3),self.nOutputPlane,self.nOutputPlane)

   self.output = torch.zeros(self.nOutputPlane,input:size(2),input:size(3))
   for i=1,input:size(2) do
      self.column_row = i
      for j=1,input:size(3) do
         self.column_column = j
         self.output[{{},i,j}] = self:forwardSingleColumn(input[{{},i,j}],B,sigma,lambda,ones_B)
      end
   end
   return self.output
end
-- encode single column to LLC 
function LLCNet:forwardSingleColumn(input,B,sigma,lambda,ones_B)
   local column_row = self.column_row
   local column_column = self.column_column
   local x = input:resize((#input)[1],1)
   -- d
   local temp1 = x*ones_B:t()-B
   local temp2 = temp1:t()*temp1
   local d = torch.diag(temp2) / (sigma)
   self.d[{column_row,column_column,{}}] = d 
   -- D
   local D = torch.diag(d)
   -- M
   self.M[{column_row,column_column,{},{}}] = (B:t()-ones_B*x:t())*(B:t()-ones_B*x:t()):t()
   -- c_tilde
   local c_tilde = (self.M[{column_row,column_column,{},{}}]+lambda*(torch.mm(D,D)))
   c_tilde = torch.inverse(c_tilde) * torch.ones((#c_tilde)[1],1)
   -- c
   local ones_c_tilde = torch.ones((#c_tilde)[1],1)
   local denominator = torch.repeatTensor((ones_c_tilde:t()*c_tilde),#c_tilde)
   local c = torch.cdiv(c_tilde,denominator)
   return c:reshape((#c)[1])
end

function LLCNet:updateGradInput(input, gradOutput)
   -- record self.gradOutput for function LLCNet:f_c_tilde(c_tilde)
   self.gradOutput = gradOutput
   -- init parameter
   local B = self.B
   local sigma = self.sigma
   local lambda = self.lambda
   local ones_B = torch.ones(self.nOutputPlane,1)
   local d = self.d
   local M = self.M

   self.gradInput = input:clone()
   self.gradInput:zero()

   for i=1,input:size(2) do
      self.column_row = i
      for j=1,input:size(3) do
         self.column_column = j
         self.gradInput[{{},i,j}] = self:backwardSingleColumn(input[{{},i,j}],d[{i,j,{}}],M[{i,j,{},{}}],B,sigma,lambda,ones_B)
      end
   end
   return self.gradInput
end

function LLCNet:backwardSingleColumn(input,d,M,B,sigma,lambda,ones_B)
   local x = input:resize((#input)[1],1)
   -- D
   local D = torch.diag(d)
   local mangrad_f_d_minor_x = self:f_d_minor_x(x,d,B,lambda,sigma,M)
   local mangrad_f_M_x = self:f_M_x(x,D,lambda,B,M)
   local mangrad = mangrad_f_d_minor_x + mangrad_f_M_x
   return mangrad:reshape((#mangrad)[1])
end

function LLCNet:f_d_minor_x(x,d,B,lambda,sigma,M)
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

function LLCNet:f_d_minor(d,M,lambda)
   local D = torch.diag(d)
   local mangrad_f_D = self:f_D(D,M,lambda)
   local mangrad = torch.diag(mangrad_f_D)
   return mangrad
end

function LLCNet:f_D(D,M,lambda)
   -- f_c_tilde
   local D_tilde = (M + lambda*(torch.mm(D,D)))
   local A = -lambda * (torch.inverse(D_tilde))
   local c_tilde = torch.inverse(D_tilde) * torch.ones((#D_tilde)[1],1)
   local B = c_tilde
   local mangrad_f_c_tilde = self:f_c_tilde(c_tilde)
   local mangrad = (D*B*mangrad_f_c_tilde:t()*A + B*mangrad_f_c_tilde:t()*A*D):t()
   return mangrad
end

function LLCNet:f_M_x(x,D,lambda,B,M)
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

function LLCNet:f_M(M,D,lambda)
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

function LLCNet:f_c_tilde(c_tilde)
  local column_row = self.column_row
  local column_column = self.column_column
  -- c_c_tilde
  local mangrad_c_c_tilde = self:c_c_tilde(c_tilde)
  -- f_c
  local ones_c_tilde = torch.ones((#c_tilde)[1],1)
  local c = c_tilde / (ones_c_tilde:t()*c_tilde)[1][1]
  local mangrad_f_c = self.gradOutput[{{},column_row,column_column}]
  -- f_c_tilde
  local mangrad = torch.zeros((#c_tilde)[1],1)
  for i = 1,(#mangrad_c_c_tilde)[2] do
     for j = 1,(#mangrad_c_c_tilde)[1] do
        mangrad[i] = mangrad[i] + mangrad_f_c[j]*mangrad_c_c_tilde[j][i]
     end
  end
  return mangrad
end

function LLCNet:c_c_tilde(c_tilde)
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


