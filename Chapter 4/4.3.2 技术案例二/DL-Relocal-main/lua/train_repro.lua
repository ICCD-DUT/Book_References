require 'nn'
require 'cunn'
require 'optim'
require 'cudnn'

-- general parameters
storeCounter = 0 -- counts parameter updates

-- parameters of pretraining
storeInterval = 1000 		-- storing snapshot after x updates
lrInit = 0.0001 		-- initial learning rate
lrInterval = 50000 		-- cutting learning rate in half after x updates
lrIntervalInit = 100000 	-- number if initial iteration without learning rate cutting
gradClamp = 0.5 		-- maximum gradient magnitude (reprojection opt. only)

oFileInit = 'obj_model_fcn_init.net'
oFileRepro = 'obj_model_fcn_repro.net'

mean = {127, 127, 127} 

dofile('MyL1Criterion.lua')

function loadModel(f, inW, inH, outW, outH)

  inputWidth = inW
  inputHeight = inH
  outputWidth = outW
  outputHeight = outH

  print('TORCH: Loading network from file: ' .. f)

  modeltemp=torch.load("mydensenet-201_1-29_repro.t7")--zhong duan shi jia zai.              
  -- -- 640 x 480
  model2 = modeltemp:cuda()

  model = torch.load(f)
  model = model:cuda()
  cudnn.convert(model, cudnn)
  cudnn.convert(model2, cudnn)

  model:evaluate()
  --model2:evaluate()

  criterion = nn.MyL1Criterion()
  criterion = criterion:cuda()

  params, gradParams = model:getParameters()
  optimState = {learningRate = lrInit}
end

function setEvaluate()
    model:evaluate()
    print('TORCH: Set model to evaluation mode.')
end

function setTraining()
    model:training()
    print('TORCH: Set model to training mode.')
end

function setGPU(i)
   cutorch.setDevice(i)
   print('TORCH: Set GPU to ' .. i)
end

function setStoreCounter(i)
  storeCounter = i
  print('TORCH: Set storeCounter to ' .. i)
end

function forward(count, data)

  local input = torch.FloatTensor(data):reshape(count, 3, inputHeight, inputWidth);
  input = input:cuda()

  -- normalize data
  for c=1,3 do
    input[{ {}, {c}, {}, {}  }]:add(-mean[c]) 
  end

  print('TORCH: Doing a forward pass.')
  input=model2:forward(input)
  local results = model:forward(input)
  results = results:reshape(3, outputHeight * outputWidth):transpose(1,2)
  results = results:double()

  local resultsR = {}
  for i = 1,results:size(1) do
    for j = 1,3 do
      local idx = (i-1) * 3 + j
      resultsR[idx] = results[{i, j}]
    end
  end

  return resultsR
end


function backward(count, loss, data, gradients)

  print('TORCH: Doing a backward pass.')
  local input = torch.FloatTensor(data):reshape(1, 3, inputHeight, inputWidth)
  local dloss_dpred = torch.FloatTensor(gradients):reshape(count, 3):transpose(1,2):reshape(1, 3, outputHeight, outputWidth)

  input = input:cuda()
  dloss_dpred = dloss_dpred:cuda()

  dloss_dpred:clamp(-gradClamp,gradClamp)

  -- normalize data
  for c=1,3 do
    input[{ {}, {c}, {}, {}  }]:add(-mean[c]) 
  end

  gradParams:zero()
  input=model2:forward(input)
  local function feval(params)
    model:backward(input, dloss_dpred)
    return loss,gradParams
  end
  optim.adam(feval, params, optimState)

  storeCounter = storeCounter + 1

  if (storeCounter % storeInterval) == 0 then
    print('TORCH: Storing a snapshot of the network.')
    model:clearState()
    torch.save(oFileRepro .. storeCounter, model)
    torch.save("mydensenet-201_1-29_repro.t7", model2)
  end

  if storeCounter > (lrIntervalInit - 1) and (storeCounter % lrInterval) == 0 then
    print('TORCH: Cutting learningrate by half. Is now: ' .. optimState.learningRate)
    optimState.learningRate = optimState.learningRate * 0.5

  end
end

