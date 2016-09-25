require 'nn'
require 'mylayer'
require 'torch'
require 'LLCNet'

outputsize = 1
hidden1 = 2
hidden2 = 5
codelength = 5
model = nn.Sequential()
model:add(nn.SpatialConvolution(1, hidden1, 3, 3))
model:add(nn.ReLU())
model:add(nn.SpatialConvolution(hidden1, hidden2, 3, 3))
model:add(nn.ReLU())
model:add(nn.LLCNet(hidden2,codelength))
model:add(nn.Reshape(codelength*4))
model:add(nn.Linear(codelength*4,outputsize))
criterion = nn.MSECriterion()

function test()
   print(model:get(2).gradInput)
   input = torch.ones(1,6,6)
   print(model:forward(input))
   input = input*(-1)
   print(model:forward(input))
end

test()
a = torch.zeros(1)
for i = 2000,2000 do
  -- random sample
  local input = (torch.rand(1,6,6)-0.5)*2     -- normally distributed example in 2d
  local output= torch.Tensor(outputsize);
  if torch.sum(input) > 0 then  -- calculate label for XOR function
    output:fill(-1)
  else
    output:fill(1)
  end

  -- feed it to the neural network and the criterion
  criterion:forward(model:forward(input), output)
  a = model.output:ne(model.output)
  if(a[1] == 1) then
      print(i)
      test()
      break
  end

  -- train over this example in 3 steps
  -- (1) zero the accumulation of the gradients
  model:zeroGradParameters()
  -- (2) accumulate gradients
  model:backward(input, criterion:backward(model.output, output))
  -- (3) update parameters with a 0.01 learning rate
  model:updateParameters(1)
end



