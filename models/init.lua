--
--  Copyright (c) 2016, Computer Vision Lab @ Seoul National University.
--  All rights reserved.
--
--
--  Model creating code

require 'nn'
require 'cunn'
require 'cudnn'
require 'tvnorm-nn'
require 'stn'
require 'nngraph'

local M = {}

function M.setup(opt, checkpoint)
    local model

    if checkpoint then
        local modelPath = paths.concat(opt.resume, checkpoint.modelFile)
        assert(paths.filep(modelPath), 'Saved model not found: ' .. modelPath)
        print('=> Resuming model from ' .. modelPath)
        model   = torch.load(modelPath):cuda()
    elseif opt.retrain ~= 'none' then
        assert(paths.filep(modelPath), 'Model not found: ' .. opt.retrain)
        print('=> Loading model from ' .. opt.retrain)
        model   = torch.load(opt.retrain):cuda()
    else
        print('=> Creating model from: models/' .. opt.networkType .. '.lua')
        model = require('models/' .. opt.networkType)(opt)
    end

   -- Set the CUDNN flags
   if opt.cudnn == 'fastest' then
      cudnn.fastest = true
      cudnn.benchmark = true
   elseif opt.cudnn == 'deterministic' then
      -- Use a deterministic convolution implementation
      model:apply(function(m)
         if m.setMode then m:setMode(1, 1, 1) end
      end)
   end    

   -- Wrap the model with DataParallelTable, if using more than one GPU
   if opt.nGPU > 1 then
      local gpus = torch.range(1, opt.nGPU):totable()
      local fastest, benchmark = cudnn.fastest, cudnn.benchmark

      local dpt = nn.DataParallelTable(1, true, true)
         :add(model, gpus)
         :threads(function()
            local cudnn = require 'cudnn'
            cudnn.fastest, cudnn.benchmark = fastest, benchmark
         end)
      dpt.gradInput = nil

      model = dpt:cuda()
   end

   local imgH   = 0
   local imgW   = 0

   if opt.dataset == "brainMRI" then
      imgH = 384
      imgW = 384
   end


   local criterion  = nn.MSECriterion():cuda()

   model:cuda()
   cudnn.convert(model, cudnn)

   return model, criterion

end

return M
