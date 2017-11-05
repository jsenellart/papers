local torch = require('torch')
local cutorch
local nn = require('nn')
require('nngraph')

local cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('WORD TRANSLATION WITHOUT PARALLEL DATA')
cmd:text()
cmd:text('Options')
cmd:option('-seed',123,'initial random seed')
cmd:option('-gpuid',0,'use cuda')
cmd:option('-srcemb','','path to source embedding')
cmd:option('-tgtemb','','path to target embedding')
cmd:option('-vocsize', 200000, 'vocabulary size')
cmd:text()

local opt = cmd:parse(arg)

if opt.gpuid > 0 then
  print('loading CUDA')
  cutorch = require('cutorch')
  require('cunn')
  cutorch.setDevice(opt.gpuid)
end

local dim = 300
-- hiiden layer size of discriminator
local hidden = 2048
-- batch size
local batchSize = 64
-- discriminant dropout
local disc_dropout = 0.1
-- smoothing label parameter
local smoothing = 0.2
-- number of iterations for discriminator training
local k = 1
-- learning rate and decay
local learningRate = 0.1
local decay = 0.99

local beta = 0.01

torch.manualSeed(opt.seed)

local function find_knn(t, v, topk)
  local d = torch.Tensor(v:size(1))
  if opt.gpuid then d = d:cuda() end
  for i = 1, v:size(1) do
    d[i] = -t:dot(v[i])/v[i]:norm()
  end
  return torch.topk(d, topk)
end

-- read embedding from text file or from .t7
local function read_embed(filename)
  print("read embedding from "..filename.." - opt.vocsize ="..opt.vocsize)
  if filename:sub(-3) == ".t7" then
    local voc, weights = table.unpack(torch.load(filename))
    assert(#voc == opt.vocsize)
    assert(weights:size(2) == dim)
    return voc, weights
  else
    local f = io.open(filename, "r")
    local header = f:read()
    local splitHeader = header:split(' ')
    assert(#splitHeader==2, "incorrect file format - header should be '#vocab dim'")
    local numWords = tonumber(splitHeader[1])
    local embeddingSize = tonumber(splitHeader[2])
    assert(numWords>=opt.vocsize, "opt.vocsize larger than vocabulary in embedding")
    assert(embeddingSize==dim, "embedding size does not match dim")
    local weights = torch.Tensor(opt.vocsize, embeddingSize)
    local voc = {}
    for i=1, opt.vocsize do
      local line = f:read()
      local splitLine = line:split(' ')
      assert(#splitLine == dim+1, "incorrect embedding format")
      table.insert(voc,splitLine[1])
      for j = 2, #splitLine do
        weights[i][j-1] = tonumber(splitLine[j])
      end
    end
    torch.save(filename.."_"..opt.vocsize..".t7", { voc, weights })
    print("  * saved to "..filename.."_"..opt.vocsize..".t7")
    return voc, weights
  end
end

local svoc, semb = read_embed(opt.srcemb)
local tvoc, temb = read_embed(opt.tgtemb)

-- the generator - input source embedding, output projection in target embedding
-- no bias
local x = nn.Identity()()
local generator = nn.gModule({x},{nn.Linear(dim, dim, false)(x)})

-- the discriminator - outputs log probability of input to be source or target
x = nn.Identity()()
local h1 = nn.Linear(dim, hidden)(nn.Dropout(disc_dropout)(x))
local o = nn.Linear(hidden,1)(nn.LeakyReLU()(h1))
local discriminator = nn.gModule({x},{nn.Sigmoid()(o)})

-- use cross entropy
local criterion = nn.BCECriterion()

local zeroClass = torch.Tensor(batchSize):fill(0)
local oneClass = torch.Tensor(batchSize):fill(1)
local smoothedOneClass = torch.Tensor(batchSize):fill(1-smoothing)

if opt.gpuid > 0 then
  generator = generator:cuda()
  discriminator = discriminator:cuda()
  zeroClass = zeroClass:cuda()
  oneClass = oneClass:cuda()
  smoothedOneClass = smoothedOneClass:cuda()
  criterion = criterion:cuda()
  semb = semb:cuda()
  temb = temb:cuda()
end

local W = generator:getParameters():reshape(dim,dim)

for iter = 1, 100 do
  local genLoss = 0
  local discLoss = 0

  for _ = 1, opt.vocsize/batchSize do
    for _ = 1, k do
      local bsrcIdx = (torch.rand(batchSize)*opt.vocsize+1):long()
      local btgtIdx = (torch.rand(batchSize)*opt.vocsize+1):long()
      local batch_src = semb:index(1, bsrcIdx)
      local batch_tgt = temb:index(1, btgtIdx)

      -- projection of source in target
      local projectedSrc = generator:forward(batch_src)

      discriminator:zeroGradParameters()

      -- calculate loss for batch src projected in target
      local discProjSrc = discriminator:forward(projectedSrc)
      discLoss = discLoss + criterion:forward(discProjSrc, zeroClass)
      discriminator:backward(projectedSrc, criterion:backward(discProjSrc, zeroClass))

      -- loss for tgt classified with smoothed label
      local discTgt = discriminator:forward(batch_tgt)
      discLoss = discLoss + criterion:forward(discTgt, smoothedOneClass)
      discriminator:backward(batch_tgt, criterion:backward(discTgt, smoothedOneClass))

      discriminator:updateParameters(learningRate)

    end

    local bsrcIdx = (torch.rand(batchSize)*opt.vocsize+1):long()
    local batch_src = semb:index(1, bsrcIdx)

    if opt.gpuid > 0 then
      batch_src = batch_src:cuda()
    end

    -- calculate loss for batch src projected in target
    local projectedSrc = generator:forward(batch_src)
    local discProjSrc = discriminator:forward(projectedSrc)

    genLoss = genLoss + criterion:forward(discProjSrc, oneClass)
    generator:zeroGradParameters()
    local gradGen = discriminator:backward(projectedSrc, criterion:backward(discProjSrc, oneClass))
    generator:backward(batch_src, gradGen)
    generator:updateParameters(learningRate)

    -- update to keep W orthogonal
    W = (1+beta) * W
    local prod = torch.mm(W, W:t())
    W:addmm(-beta,prod,W)
  end

  print('--- ',iter,'genLoss='..genLoss*batchSize/opt.vocsize, 'discLoss='..discLoss*batchSize/opt.vocsize/k,
        'learningRate='..learningRate)

  learningRate = learningRate * decay

end


for i = 1, 10000 do
  local projSEmb = generator:forward(semb[i])
  local y, idx = find_knn(projSEmb, temb, 10)
  print('* '..svoc[i])
  for j = 1, idx:size(1) do
    print('  '..tvoc[idx[j]], y[j])
  end
end
