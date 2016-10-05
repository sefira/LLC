feature_data_forSave = {}
feature_data_forSaveFilename = "../res/deep_cnn_feature.dat"

function setforSaveFilename(filename)
	feature_data_forSaveFilename = filename
end
   
function append(feature_data)
	feature_data_forSave[#feature_data_forSave + 1] = feature_data
end

function clearSaveData ()
	feature_data_forSave = {}
end

function saveData ()
	torch.save(feature_data_forSaveFilename, feature_data_forSave)
end
