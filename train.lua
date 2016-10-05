--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The training loop and learning rate schedule
--

local optim = require 'optim'

local M = {}
local Trainer = torch.class('USOF.Trainer', M)

function Trainer:__init(model, criterion, opt, optimState)
   self.model           = model
   self.criterion       = criterion
   self.optimState      = optimState or {
      learningRate      = opt.learningRate,
      learningRateDecay = 0.0,
      momentum          = opt.momentum,
      nesterov          = true,
      dampening         = 0.0,
      weightDecay       = opt.weightDecay,
      beta1             = opt.beta_1,
      beta2             = opt.beta_2,
   }
   self.opt = opt
   self.params, self.gradParams = model:getParameters()
end

function Trainer:train(epoch, dataloader)
   -- Trains the model for a single epoch
   self.optimState.learningRate = self:learningRate(epoch)
   local timer              = torch.Timer()
   local dataTimer          = torch.Timer()

   local function feval()
    --   return self.criterion.output, self.gradParams
      return self.criterion.output, self.gradParams
   end

   local losses     = {}
   local trainSize  = dataloader:size()
   local lossSum    = 0.0
   local debug_loss = 0.0
   local N          = 0

   print('=============================')
   print(self.optimState)
   print('=============================')
   
   print('=> Training epoch # ' .. epoch)
   -- set the batch norm to training mode
   self.model:training()
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      local output      = self.model:forward(self.input)
      local batchSize   = output:size(1)
      
      -- normalizing outputs 
      -- self.model.output
      
      local loss        = self.criterion:forward(self.model.output, self.target)

      self.model:zeroGradParameters()
      self.criterion:backward(self.model.output, self.target)
      self.model:backward(self.input, self.criterion.gradInput)

      local garbage, tmp_loss = 0,0
      if self.opt.optimizer == 'sgd' then
        garbage, tmp_loss = optim.sgd(feval, self.params, self.optimState)
      elseif self.opt.optimizer == 'adam' then
        garbage, tmp_loss = optim.adam(feval, self.params, self.optimState)
      elseif self.opt.optimizer == 'adagrad' then
        garbage, tmp_loss = optim.adagrad(feval, self.params, self.optimState)
      end

      N = N + batchSize
      lossSum = lossSum + loss
      debug_loss = debug_loss + loss
      if (n%200) == 0 then
          losses[#losses + 1] = debug_loss/200

          gnuplot.pngfigure('trainSpecs/trainLoss_' .. tostring(epoch) .. '.png')

          gnuplot.plot({ torch.range(1, #losses), torch.Tensor(losses), '-' })
          gnuplot.plotflush()
          debug_loss = 0.0

          print(string.format('Gradient min: %1.4f \t max:  %1.4f \t norm: %1.4f', torch.min(self.gradParams:float()), torch.max(self.gradParams:float()), torch.norm(self.gradParams:float())))
    
          image.save('current_ref.png', self.input[{ {1},{1},{},{} }]:reshape(1,self.input[{ {1},{1},{},{} }]:size(3),self.input[{ {1},{1},{},{} }]:size(4)))
          image.save('current_tar.png', self.input[{ {1},{2},{},{} }]:reshape(1,self.input[{ {1},{2},{},{} }]:size(3),self.input[{ {1},{2},{},{} }]:size(4)))
          
          -- print(self.target:size())
          -- print(output:size())
          print('============================================================================')
          print(('|  Predicted Matrix: [%2.4f  %2.4f  %2.4f]     GroundTruth Matrix: [%2.4f  %2.4f  %2.4f]'):format(output[1][1][1],output[1][1][2],output[1][1][3],self.target[1][1][1],self.target[1][1][2],self.target[1][1][3]))
          print(('|                    [%2.4f  %2.4f  %2.4f]                         [%2.4f  %2.4f  %2.4f]'):format(output[1][2][1],output[1][2][2],output[1][2][3],self.target[1][2][1],self.target[1][2][2],self.target[1][2][3]))
          print('============================================================================')
          collectgarbage()
      end

	if (n%50) == 0 then
   	     print((' | Epoch: [%d][%d/%d]    Time %.3f  Data %.3f  loss %1.4f'):format(
             epoch, n, trainSize, timer:time().real, dataTime, loss))--total_loss))
   	   -- check that the storage didn't get changed due to an unfortunate getParameters call
   	end

 	assert(self.params:storage() == self.model:parameters()[1]:storage())

      timer:reset()
      dataTimer:reset()
   end

--    return top1Sum / N, top5Sum / N, lossSum / N
    return lossSum / N
end

function Trainer:test(epoch, dataloader)

   local timer    = torch.Timer()
   local size     = dataloader:size()
   local N        = 0
   local lossSum  = 0.0
   local debug_loss = 0.0
   local losses   = {}
   local param_11, param_12, param_13, param_21, param_22, param_23 = 0.0,0.0,0.0,0.0,0.0,0.0
   
   self.model:evaluate()
   
   for n, sample in dataloader:run() do
      -- Copy input and target to the GPU
      self:copyInputs(sample)

      local output      = self.model:forward(self.input)
      local batchSize   = output:size(1)
      local loss        = self.criterion:forward(self.model.output, self.target)

      for b_iter =1, self.opt.batchSize do
          if output[b_iter][1][1] == self.target[b_iter][1][1] then param_11 = param_11 + 1 end
          if output[b_iter][1][2] == self.target[b_iter][1][2] then param_12 = param_12 + 1 end
          if output[b_iter][1][3] == self.target[b_iter][1][3] then param_13 = param_13 + 1 end
          if output[b_iter][2][1] == self.target[b_iter][2][1] then param_21 = param_21 + 1 end
          if output[b_iter][2][2] == self.target[b_iter][2][2] then param_22 = param_22 + 1 end
          if output[b_iter][2][3] == self.target[b_iter][2][3] then param_23 = param_23 + 1 end
      end

      N           = N + batchSize
      lossSum     = lossSum + loss
      debug_loss  = debug_loss + loss
      if (n%10) == 0 then
        print((' | Test: [%d][%d/%d]    Time %.3f  loss %1.4f'):format( epoch, n, size, timer:time().real, loss))
        losses[#losses + 1] = debug_loss/10
        gnuplot.pngfigure('trainSpecs/testLoss_' .. tostring(epoch) .. '.png')
        gnuplot.plot({ torch.range(1, #losses), torch.Tensor(losses), '-' })
        gnuplot.plotflush()
        debug_loss = 0.0
        if (n%100) == 0 then
            print('============================================================================')
            print(('|  Accuracy: [%2.4f%%  %2.4f%%  %2.4f%%]'):format(param_11/N*100, param_12/N*100, param_13/N*100))
            print(('|            [%2.4f%%  %2.4f%%  %2.4f%%]'):format(param_21/N*100, param_22/N*100, param_23/N*100))
            print('============================================================================')
        end
      end

      timer:reset()
   end
   self.model:training()

--   print((' * Finished epoch # %d     loss: %1.4f\n'):format(
--      epoch, lossSum / N))

   return lossSum / N
end

function Trainer:copyInputs(sample)
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
   self.input = self.input or (self.opt.nGPU == 1
      and torch.CudaTensor()
      or cutorch.createCudaHostTensor())
   self.target = self.target or torch.CudaTensor()

   self.input:resize(sample.input:size()):copy(sample.input)
   self.target:resize(sample.target:size()):copy(sample.target)
end

function Trainer:learningRate(epoch)
   -- Training schedule
   local decay = 0
   if (self.opt.dataset == 'ABIDE') and (epoch%10 == 0) then
	return self.optimState.learningRate/2
   else
	return self.optimState.learningRate
   end 

end

return M.Trainer
