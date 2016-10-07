feature_data_forSave = {}
feature_data_forSaveFilename = "../res/deep_cnn_feature.dat"

-- some function for save deep feature during CNN forward
function utils.set_feature_data_filename(filename)
	feature_data_forSaveFilename = filename
end
   
function utils.append_feature_data(feature_data)
	feature_data_forSave[#feature_data_forSave + 1] = feature_data:clone()
end

function utils.clear_feature_data ()
	feature_data_forSave = {}
end

function utils.save_feature_data ()
	torch.save(feature_data_forSaveFilename, feature_data_forSave)
	print("already save deep feature data in " .. feature_data_forSaveFilename)
end

-- function for read deep feature file which already saved in ../res by above function
function utils.reshape_deepfeature_from_file(filename)
   local filename = filename or feature_data_forSaveFilename
   print("reshape deep feature from file " .. feature_data_forSaveFilename)
   local data = torch.load(feature_data_forSaveFilename)
   if #data < 2 then
      return torch.Tensor(1,1)
   end
   local data_matrix = torch.Tensor(#data*(data[1]:size(2))*(data[1]:size(3)),data[1]:size(1))
   local deep_feature_count = 1
   for n=1,#data do
      local item = data[n]
      for i=1,item:size(2) do
         for j=1,item:size(3) do
            data_matrix[{deep_feature_count,{}}] = item[{{},i,j}]:clone()
            deep_feature_count = deep_feature_count + 1
         end
      end
   end
   return data_matrix
end

-- function for save deep feature which comes from reshape_deepfeature_from_file() to csv
function utils.save_deep_feature_to_csv(filename)
   local filename = filename or feature_data_forSaveFilename
   print("save deep feature to csv " .. feature_data_forSaveFilename .. ".csv")
   local feature_data = utils.reshape_deepfeature_from_file(filename)
   local csvfile = assert(io.open(filename .. ".csv", "w")) -- open a file for serialization

   splitter = ","
   for i=1,feature_data:size(1) do
       for j=1,feature_data:size(2) do
           csvfile:write(feature_data[i][j])
           if j == feature_data:size(2) then
               csvfile:write("\n")
           else
               csvfile:write(splitter)
           end
       end
   end
   csvfile:close()
end

-- function for read centroids.csv
function utils.read_centroids(filename)
   csv2tensor = require 'csv2tensor'
   local filename = filename or "../res/centroids.csv"
   training_tensor, column_names = csv2tensor.load(filename)
   
   return training_tensor, column_names   
end



