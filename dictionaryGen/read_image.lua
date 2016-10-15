--------------------------------
--TODO transforms tensor to cuda
--------------------------------

require 'torch'   -- torch
require 'image'   -- for color transforms
require 'nn'      -- provides a normalization operator
require 'csvigo'  -- load csv data set file

----------------------------------------------------------------------

--see if the file exists
function file_exists(file)
    local f = io.open(file, "rb")
    if f then 
        f:close()
    end
    return f ~= nil
end

function read_file (file)
    if not file_exists(file) then 
        return {} 
    end
    lines = csv.load({ path = file, header = "false", mode = "raw"})
    header = table.remove(lines, 1)
    return lines
end

-- Preprocess an image before passing it to a Caffe model.
-- We need to rescale from [0, 1] to [0, 255], convert from RGB to BGR,
-- and subtract the mean pixel.
function preprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):mul(256.0)
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img:add(-1, mean_pixel)
  return img
end

image_size = 224
-- read train data. iterate train.txt
train_csv = read_file("../data/train.csv")
train_data = {}
for i = 1, #train_csv do
    local res = {}
    --s = "data1/buxingyuan12/1/2_1.jpg 1"
    for v in string.gmatch(train_csv[i], "[^%s]+") do
        res[#res + 1] = v
    end
    filename = res[1]
    local train_labels
    if ClassNLL then
        train_labels = res[2] + 1 -- train_labels = 1 or 2
    else
        train_labels = torch.Tensor(2):zero() -- train_labels = 01 or 10
        train_labels[res[2] + 1] = 1 -- class
        if enableCuda then
            train_labels:cuda()
        else
            train_labels:float()
        end
    end
    -- here need to mul(255) due to torch will auto mul(1/255) for a jpg    
	local content_image = image.load("../data/" .. filename, 3)
	content_image = image.scale(content_image, image_size, 'bilinear')
	local imageread = preprocess(content_image)
    --print(imageread:max())
    local train_image = imageread
    local train_data_temp
    if enableCuda then
        train_data_temp = {
            data = train_image:cuda(),
            labels = train_labels
        }
    else        
        train_data_temp = {
            data = train_image:float(),
            labels = train_labels
        }
    end
    train_data[#train_data + 1] = train_data_temp
    if(i % 100 == 0) then
        print("train data: " .. i)
    end
end

-- read test data. iterate test.txt
test_csv = read_file("../data/test.csv")
test_data = {}
for i = 1, #test_csv do
    local res = {}
    --s = "data1/buxingyuan12/1/2_1.jpg 1"
    for v in string.gmatch(test_csv[i], "[^%s]+") do
        res[#res + 1] = v
    end
    filename = res[1]
    local test_labels
    if ClassNLL then
        test_labels = res[2] + 1 -- test_labels = 1 or 2
    else
        test_labels = torch.Tensor(2):zero() -- test_labels = 01 or 10
        test_labels[res[2] + 1] = 1 -- class
        if enableCuda then
            test_labels:cuda()
        else
            test_labels:float()
        end
    end
    -- here need to mul(255) due to torch will auto mul(1/255) for a jpg    
	local content_image = image.load("../data/" .. filename, 3)
	content_image = image.scale(content_image, image_size, 'bilinear')
	local imageread = preprocess(content_image)
    --print(imageread:max())
    local test_image = imageread
    local test_data_temp
    if enableCuda then
        test_data_temp = {
            data = test_image:cuda(),
            labels = test_labels
        }
    else        
        test_data_temp = {
            data = test_image:float(),
            labels = test_labels
        }
    end
    test_data[#test_data + 1] = test_data_temp
    if(i % 100 == 0) then
        print("test data: " .. i)
    end
end

trsize = #train_csv
tesize = #test_csv

