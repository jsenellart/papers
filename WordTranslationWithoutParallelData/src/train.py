import torch
from torch.autograd import Variable
import torch.nn as nn
from scipy.spatial.distance import cosine
import progressbar
from net import Generator, Discriminator
from os import path
import numpy as np

import argparse
import math

parser = argparse.ArgumentParser(description='Word Translation Without Parallel Data')
parser.add_argument('srcemb', nargs=1, type=str, help='source word embedding')
parser.add_argument('tgtemb', nargs=1, type=str, help='target word embedding')
parser.add_argument('--seed', type=int, default=123, help='initial random seed')
parser.add_argument('--vocSize', type=int, default=200000, help='vocabulary size')
parser.add_argument('--dim', default=300, type=int, help='embedding size')
parser.add_argument('--hidden', default=2048, type=int, help='discriminator hidden layer size [3.1]')
parser.add_argument('--discDropout', default=0.1, type=float, help='discriminator dropout [3.1]')
parser.add_argument('--smoothing', default=0.2, type=float, help='label smoothing value [3.1]')
parser.add_argument('--samplingRange', default=50000, type=int, help='sampling range on vocabulary for adversarial training [3.2]')
parser.add_argument('--beta', default=0.0001, type=float, help='orthogonality adjustment parameter (equation 7)')
parser.add_argument('--k', default=1, type=int, help='#iteration of discriminator training for each iteration')
parser.add_argument('--batchSize', default=64, type=int, help='batch size')
parser.add_argument('--learningRate', default=0.1, type=float, help='learning rate')
parser.add_argument('--decayRate', default=0.99, type=float, help='decay rate')
parser.add_argument('--nEpochs', default=100, type=int, help='number of epochs')
parser.add_argument('--halfDecayThreshold', default=0.03, type=float, help='if valid relative increase > this value for 2 epochs, half the LR')
parser.add_argument('--halfDecayDelay', default=8, type=int, help='if no progress in this period, half the LR')
parser.add_argument('--knn', default=10, type=int, help='number of neighbors to extract')
parser.add_argument('--refinementIterations', default=1, type=int, help='number of iteration of refinement')
parser.add_argument('--distance', type=str, default='CSLS', help='distance to use NN or CSLS [2.3]', choices=['CSLS', 'NN'])
parser.add_argument('--validDistance', type=str, default='COS', help='validation distance', choices=['CSLS', 'COS'])
parser.add_argument('--load', type=str, help='load parameters of generator')
parser.add_argument('--save', type=str, help='save parameters of generator', required=True)
parser.add_argument('--dump_output', type=str, help='dump the complete mapped dictionary')
parser.add_argument('--evalDict', type=str, help='dictionary for evaluation')
parser.add_argument('--gpuid', default=-1, type=int)

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

if args.gpuid >= 0:
  # allocate dummy tensor to check GPU is ok
  with torch.cuda.device(args.gpuid):
    torch.Tensor(1).cuda()

print("* params: ", args)

# -------------------------------------------------------
# READ DICTIONARY

evalDict = {}
def read_dict(filename):
  with open(filename) as f:
    for line in f:
      lineSplit = line.strip().split("\t")
      assert len(lineSplit)==2, "invalid format in dictionary"
      if not lineSplit[0] in evalDict:
        evalDict[lineSplit[0]] = [lineSplit[1]]
      else:
        evalDict[lineSplit[0]].append(lineSplit[1])

# check an entry meaning and returns @1, @5, @10
def eval_entry(src, tgts):
  if not src in evalDict:
    return
  meanings = evalDict[src]
  for i in range(min(len(tgts), 10)):
    if tgts[i] in meanings:
      if i == 0: return (1, 1, 1)
      if i < 5: return (0, 1, 1)
      return (0, 0, 1)
  return (0, 0, 0)

def eval_dictionary(d):
  s = [0, 0, 0]
  c = 0
  for k in d.keys():
    score = eval_entry(k, d[k])
    if score:
      c += 1
      s = [x+y for x,y in zip(s,score)]
  s = [ int(x/c*10000.)/100 for x in s ]
  return s

if args.evalDict:
  print("* read "+args.evalDict+" dictionary for evaluation")
  read_dict(args.evalDict)
  print("  => ", len(evalDict.keys()), "entries")

# -------------------------------------------------------
# READ EMBEDDING

def read_embed(filename):
  print("* read embedding from "+filename+" - args.vocSize="+str(args.vocSize))
  if filename[-4:] == '.bin':
    emb = torch.load(filename)
    return emb[0], emb[1]
  else:
    with open(filename) as f:
      header = f.readline().strip()
      headerSplit = header.split(" ")
      numWords = int(headerSplit[0])
      embeddingSize = int(headerSplit[1])
      assert len(headerSplit)==2, "incorrect file format - header should be '#vocab dim'"
      assert numWords>=args.vocSize, "args.vocSize larger than vocabulary in embedding"
      assert embeddingSize == args.dim, "embedding size does not match dim"
      weights = torch.Tensor(args.vocSize, embeddingSize)
      voc = []
      vocDict = dict()
      i = 0
      bar = progressbar.ProgressBar(max_value=args.vocSize)
      while i != args.vocSize:
        line = f.readline().strip()
        splitLine = line.split(" ")
        if len(splitLine) == args.dim + 1:
          token = splitLine[0]
          if token in vocDict:
            print('*** duplicate key in word embedding: '+token)
          else:
            vocDict[token] = i
            voc.append(token)
            for j in range(1, args.dim):
              weights[i][j-1] = float(splitLine[j])
            bar.update(i)
            i = i + 1
      torch.save([voc, weights], filename+"_"+str(args.vocSize)+".bin")
      print("  * saved to "+filename+"_"+str(args.vocSize)+".bin")
      return voc, weights

svoc, semb = read_embed(args.srcemb[0])
tvoc, temb = read_embed(args.tgtemb[0])

# -------------------------------------------------------
# PAIR MATCHING

# initialize index using FAISS

import faiss
print("* indexing target vocabulary with FAISS")
# index the target embedding
index = faiss.IndexFlatL2(args.dim)
index.add(temb.numpy())

# given a tensor or a batch of tensor returns distance and index to closes target neighbours
def NN(v):
  cv = v
  if v.dim() == 1:
    cv.resize_(1, cv.shape[0])
  D, I=index.search(cv.numpy(), args.knn)
  return D, I, D

# calculate rs on the full vocabulary or load it from file
rs = None
rsfile = args.tgtemb[0]+'_rs_knn'+str(args.knn)
if path.isfile(rsfile):
  print("* read rs file from: "+rsfile)
  rs = torch.load(rsfile)
else:
  print("* preparing rs file (on vocabulary size/knn) - it will take a little while - but will get serialized for next iterations")
  bar = progressbar.ProgressBar()
  rs = torch.Tensor(args.vocSize)
  for istep in bar(range(0, args.vocSize, 500)):
    istepplus = min(istep+500, args.vocSize)
    Ds, Is, Cs = NN(temb[istep:istepplus])
    for i in range(istep, istepplus):
      rs[i] = 0
      for l in range(args.knn):
        rs[i] += cosine(temb[i].numpy(), temb[Is[i-istep][l]].numpy())
      rs[i] /= args.knn
  print("* save rs file to: "+rsfile)
  torch.save(rs, rsfile)

def CSLS(v):
  # get nearest neighbors and return adjusted cos distance
  D, I, COS = NN(v)
  COS = np.copy(D)
  for idx in range(v.shape[0]):
    rt = 0
    for j in range(args.knn):
      COS[idx][j] = cosine(v[idx].numpy(), temb[I[idx][j]].numpy())
      rt += COS[idx][j]
    rt /= args.knn
    for j in range(args.knn):
      D[idx][j] = 2*COS[idx][j]-rs[I[idx][j]]-rt
  return D, I, COS

def find_matches(v, distance):
  if distance == 'NN':
    return NN(v)
  return CSLS(v)

def get_dictionary(n, distance):
  # get the first n source vocab - and project in target embedding, find their mappings
  srcSubset = semb[0:n]
  if args.gpuid>=0:
    with torch.cuda.device(args.gpuid):
      srcSubset = srcSubset.cuda()

  proj = generator(Variable(srcSubset, requires_grad = False)).data.cpu()

  D, I, COS = find_matches(proj, distance)

  d = {}
  dID = {}

  validationScore = 0

  for i in range(0, n):
    distance = D[i].tolist()
    idx = list(range(args.knn))
    idx.sort(key=distance.__getitem__)
    if args.validDistance=='COS':
      validationScore += COS[i][idx[0]]
    else:
      validationScore += distance[idx[0]]
    dID[i] = [I[i][idx[j]] for j in range(args.knn)]
    d[svoc[i]] = [tvoc[I[i][idx[j]]] for j in range(args.knn)]

  return d, validationScore/n, dID

# -------------------------------------------------------
# MODEL BUILDING

discriminator = Discriminator(args)
generator = Generator(args)

print("* Loss Initialization")
loss_fn = nn.BCELoss()
print(loss_fn)

zeroClass = Variable(torch.Tensor(args.batchSize).fill_(0), requires_grad = False)
oneClass = Variable(torch.Tensor(args.batchSize).fill_(1), requires_grad = False)
smoothedOneClass = Variable(torch.Tensor(args.batchSize).fill_(1-args.smoothing), requires_grad = False)

if args.gpuid>=0:
  with torch.cuda.device(args.gpuid):
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    zeroClass = zeroClass.cuda()
    oneClass = oneClass.cuda()
    smoothedOneClass = smoothedOneClass.cuda()

learningRate = args.learningRate

# -------------------------------------------------------
# TRAINING

if args.nEpochs>0:
  print("* Start Training")
  valids = []
  optimalScore = 10000000
  stopCondition = False
  it = 1
  while it <= args.nEpochs and not stopCondition:
    genLoss = 0
    discLoss = 0
    print("  * Epoch", it)
    bar = progressbar.ProgressBar()
    N = min(args.samplingRange, args.vocSize)
    for i in bar(range(0, math.ceil(N/args.batchSize))):
      for j in range(0, args.k):
        bsrcIdx = torch.min((torch.rand(args.batchSize)*N).long(), torch.LongTensor([N-1]))
        btgtIdx = torch.min((torch.rand(args.batchSize)*N).long(), torch.LongTensor([N-1]))
        batch_src = Variable(torch.index_select(semb, 0, bsrcIdx))
        batch_tgt = Variable(torch.index_select(temb, 0, btgtIdx))
        if args.gpuid>=0:
          with torch.cuda.device(args.gpuid):
            batch_src = batch_src.cuda()
            batch_tgt = batch_tgt.cuda()

        # projection of source in target
        projectedSrc = generator(batch_src)

        discriminator.zero_grad()

        # calculate loss for batch src projected in target
        discProjSrc = discriminator(projectedSrc).squeeze()
        loss = loss_fn(discProjSrc, zeroClass)
        discLoss = discLoss + loss.data[0]
        loss.backward()

        # loss for tgt classified with smoothed label
        discTgt = discriminator(batch_tgt).squeeze()
        loss = loss_fn(discTgt, smoothedOneClass)
        discLoss = discLoss + loss.data[0]
        loss.backward()

        for param in discriminator.parameters():
          param.data -= learningRate * param.grad.data

      bsrcIdx = torch.min((torch.rand(args.batchSize)*N).long(), torch.LongTensor([N-1]))
      batch_src = Variable(torch.index_select(semb, 0, bsrcIdx))
      if args.gpuid>=0:
        with torch.cuda.device(args.gpuid):
          batch_src = batch_src.cuda()

      # calculate loss for batch src projected in target
      projectedSrc = generator(batch_src)
      discProjSrc = discriminator(projectedSrc).squeeze()

      generator.zero_grad()
      loss = loss_fn(discProjSrc, oneClass)
      genLoss = genLoss + loss.data[0]
      loss.backward()

      for param in generator.parameters():
        param.data -= learningRate * param.grad.data

      generator.orthogonalityUpdate(args.beta)

    evalScore = 'n/a'
    d, validationScore, dID = get_dictionary(10000, args.distance)

    if evalDict:
      evalScore = eval_dictionary(d)

    print('  * --- ',it,'genLoss=',genLoss*args.batchSize/N, 'discLoss=', discLoss*args.batchSize/N/args.k,
          'learningRate=', learningRate, 'valid=', validationScore, 'eval=', evalScore)

    valids.append(validationScore)

    if validationScore < optimalScore:
      generator.save(args.save+"_adversarial.t7")
      optimalScore = validationScore
      optimalScoreIt = it
      optimalScoreLR = learningRate
      print('  => saved as optimal W')

    # if validationScore increases than halfDecayThreshold % above optimal score
    # or no progress for args.haldDecayDelay epochs - come back to optimal and half decay
    if ((validationScore-optimalScore)/abs(optimalScore) > args.halfDecayThreshold
        or it - optimalScoreIt > args.halfDecayDelay):
      generator.load(args.save+"_adversarial.t7")
      it = optimalScoreIt
      print(' ***** HALF DECAY - go back to iteration ', it)
      learningRate = optimalScoreLR / 2
      optimalScoreLR = learningRate
    else:
      learningRate = learningRate * args.decayRate

    it += 1
    # stop completely when learningRate is not more than 20 initial learning rate
    stopCondition = learningRate < args.learningRate / 20

# -------------------------------------------------------
# EXTRACT 10000 first entries and calculate W using Procrustes solution

print('* reloading best saved')
generator.load(args.save+"_adversarial.t7")

if args.refinementIterations > 0:
  print('* Start Refining')

  evalScore = 'n/a'
  d, v, dID = get_dictionary(10000, args.distance)

  if evalDict:
    evalScore = eval_dictionary(d)

  print('  - CSLS score before refinement', v, evalScore)
  for itref in range(args.refinementIterations):
    ne = len(d.keys())
    X = np.zeros((ne, args.dim))
    Y = np.zeros((ne, args.dim))
    idx = 0
    for k in dID.keys():
      X[idx] = semb[k].numpy()
      Y[idx] = temb[dID[k][0]].numpy()
      idx = idx + 1
    A = np.matmul(Y.transpose(), X)
    U, s, V = np.linalg.svd(A, full_matrices=True)
    WP = np.matmul(U, V)
    generator.set(torch.from_numpy(WP))
    d, v, dID = get_dictionary(10000, args.distance)

    evalScore = 'n/a'
    if evalDict:
      evalScore = eval_dictionary(d)

    print('  - CSLS score after refinement iteration', v, evalScore)

  generator.save(args.save+"_refinement.t7")

# -------------------------------------------------------
# GET RESULTS

if args.dump_output:
  with open(args.dump_output, 'w') as fd:
    d, v, dID = get_dictionary(args.vocSize, args.distance)
    for k in d.keys():
      fd.write(k+"\t"+"\t".join(d[k])+"\n")
