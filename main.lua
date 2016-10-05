--
--  Copyright (c) 2016, Computer Vision Lab @ Seoul National University.
--  All rights reserved.
--
--

require 'gnuplot'
require 'torch'
require 'paths'
require 'optim'
require 'nn'

local models        = require 'models/init'
local DataLoader    = require 'dataloader'
local opts          = require 'opts'
local Trainer       = require 'train'
local checkpoints   = require 'checkpoints'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)
cutorch.setDevice(1)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)

-- Create model
local model, criterion = models.setup(opt, checkpoint)

-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)


-- for n, sample in valLoader:run() do
--     print(n)
--     print(sample)
-- end

-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt, optimState)

--------TO DO -----------TO DO----------------
if opt.testOnly then
   local loss = trainer:test(0, valLoader)
   print(string.format(' * Results loss: %1.4f  top5: %6.3f', loss))
   return
end
---------------------------------------------


--------TO DO -----------TO DO----------------
local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
for epoch = startEpoch, opt.nEpochs do
   -- Train for a single epoch
   local trainLoss  = trainer:train(epoch, trainLoader)

   local lossFile = 'trainLoss_' .. epoch .. '.t7'
   torch.save(paths.concat(opt.save, lossFile), trainLoss)

   -- Run model on validation set
   local testLoss   = trainer:test(epoch, valLoader)

   local testLossFile = 'testLoss_' .. epoch .. '.t7'
   torch.save(paths.concat(opt.save, testLossFile), testLoss)

   checkpoints.save(epoch, model, trainer.optimState, opt)
end
---------------------------------------------
