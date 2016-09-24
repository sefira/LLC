require 'nn'
require 'mylayer'
require 'torch'

model = nn.Sequential()
model:add(nn.SpatialConvolution(1, 2, 3, 3))
model:add(nn.SpatialConvolution(2, 5, 3, 3))
model:add(nn.Reshape(5))
for i = 1,100 do
  -- random sample
  local input = torch.randn(1,5,5)     -- normally distributed example in 2d
  local output= torch.Tensor(5);
  if torch.sum(input) > 0 then  -- calculate label for XOR function
    output:fill(-1)
  else
    output:fill(1)
  end

  -- feed it to the neural network and the criterion
  criterion:forward(model:forward(input), output)

  -- train over this example in 3 steps
  -- (1) zero the accumulation of the gradients
  model:zeroGradParameters()
  -- (2) accumulate gradients
  model:backward(input, criterion:backward(model.output, output))
  -- (3) update parameters with a 0.01 learning rate
  model:updateParameters(0.01)
end
