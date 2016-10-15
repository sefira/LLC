require 'nn'
require 'torch'
require 'LLCNet'
require 'utils'

if useVGG then
    require 'loadcaffe'
    local proto_file = 'models/VGG_ILSVRC_19_layers_deploy.prototxt'
    local model_file = 'models/VGG_ILSVRC_19_layers.caffemodel'
    local loadcaffe_backend = 'nn'
    model = loadcaffe.load(proto_file, model_file, loadcaffe_backend):float()
else
  model = nn.Sequential()                  -- intend to simulate VGG output size
  model:add(nn.SpatialConvolution(3, 512, 3, 3)) -- 224 * 224 * 512
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(2,2,2,2)) -- 112 * 112 * 512
  model:add(nn.SpatialMaxPooling(2,2,2,2)) -- 56 * 56 * 512
  model:add(nn.SpatialMaxPooling(2,2,2,2)) -- 28 * 28 * 512
  model:add(nn.SpatialMaxPooling(2,2,2,2)) -- 14 * 14 * 512
  model:add(nn.SpatialMaxPooling(2,2,2,2)) -- 7 * 7 * 512
  model:float()

function test1()
   utils.clear_feature_data ()
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