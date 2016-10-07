require 'nn'
require 'mylayer'
require 'torch'
require 'LLCNet'
require 'utils'
timer = torch.Timer()

outputsize = 1
hidden1 = 2
hidden2 = 5
codelength = 5
inputsize = 6
columnsize = (inputsize == 5) and 1 or 4
model = nn.Sequential()
model:add(nn.SpatialConvolution(1, hidden1, 3, 3))
model:add(nn.ReLU())
model:add(nn.SpatialConvolution(hidden1, hidden2, 3, 3))
model:add(nn.ReLU())
model:add(nn.LLCNet(hidden2,codelength))
model:add(nn.Reshape(codelength*columnsize))
model:add(nn.Linear(codelength*columnsize,outputsize))
criterion = nn.MSECriterion()

function test0()
   --print(model:get(7).gradInput)
   -- target = -1
   input = torch.ones(1,inputsize,inputsize)
   print(model:forward(input))
   -- target = 1
   input = input*(-1)
   print(model:forward(input))
end

function test1()
   testnum = 5
   for i = 1,testnum do 
     local input = (torch.rand(1,inputsize,inputsize)-0.5)*2
     local output= torch.Tensor(outputsize);
     if torch.sum(input) > 0 then  -- calculate label for XOR function
       output:fill(-1)
     else
       output:fill(1)
     end

     print(output[1])
     print(model:forward(input))
     utils.append_feature_data(model:get(4).output)
   end
   utils.save_feature_data()
   
   deepfeature = torch.load(feature_data_forSaveFilename)
   modelmirror = nn.Sequential()
   modelmirror:add(model:get(5))
   modelmirror:add(model:get(6))
   modelmirror:add(model:get(7))
   for i = 1,testnum do 
      print(modelmirror:forward(deepfeature[i]))
   end

end

test0()
a = torch.zeros(1)
for i = 1999,2000 do  
print(i)  
  -- random sample
  local input = (torch.rand(1,inputsize,inputsize)-0.5)*2     -- normally distributed example in 2d
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
  --dofile 'testscript.lua'
  -- (3) update parameters with a 0.01 learning rate
  model:updateParameters(0.01)
end

--test0()

print('Time elapsed for LLC: ' .. timer:time().real .. ' seconds')
