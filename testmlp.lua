require 'torch'
require 'nn'
require 'mylayer'

mlp = nn.Sequential();  -- make a multi-layer perceptron
inputs = 2; outputs = 1; HUs1 = 5; Hus2 = 10-- parameters
B = torch.rand(HUs1,Hus2)
-- print(B)
mlp:add(nn.Linear(inputs, HUs1))
mlp:add(nn.LLC(B))
mlp:add(nn.Linear(Hus2, outputs))
criterion = nn.MSECriterion()

for i = 1,2500 do
  -- random sample
  local input= torch.randn(inputs);     -- normally distributed example in 2d
  local output= torch.Tensor(1);
  if input[1]*input[2] > 0 then  -- calculate label for XOR function
    output[1] = -1
  else
    output[1] = 1
  end

  -- feed it to the neural network and the criterion
  criterion:forward(mlp:forward(input), output)

  -- train over this example in 3 steps
  -- (1) zero the accumulation of the gradients
  mlp:zeroGradParameters()
  -- (2) accumulate gradients
  mlp:backward(input, criterion:backward(mlp.output, output))
  -- (3) update parameters with a 0.01 learning rate
  mlp:updateParameters(0.1)
end
