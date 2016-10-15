--------------------------------
--TODO transforms tensor to cuda
--------------------------------
require 'torch'
----------------------------------------------------------------------
----------------------------------------------------------------------

print '==> executing all'

-------------------configuration------------------
enableCuda = false -- use cuda
useVGG = false -- load VGG model

if enableCuda then
    print "CUDA enable"
    require 'cunn'
    require 'cutorch'
end
-------------------configuration------------------
dofile 'read_image.lua'
dofile 'extract_deep_feature.lua'
