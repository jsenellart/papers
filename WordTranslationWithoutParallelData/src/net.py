import torch
from torch.autograd import Variable
import torch.nn as nn


class Generator(nn.Module):
  def __init__(self, args):
    super(Generator, self).__init__()

    print("* Generator Model Initialization")
    self.net = nn.Linear(args.dim, args.dim, False)

    if args.load:
      print("  * Load parameters from file: "+args.load)
      self.net.load_state_dict(torch.load(args.load))

    # get the mapping and initialize
    for param in self.net.parameters():
      if not args.load:
        param.data.uniform_(-0.1,0.1)
      self.W = param.data

    print(self.net)

  def load(self, filename):
    self.net.load_state_dict(torch.load(filename))

  def forward(self, x):
    return self.net.forward(x)

  def save(self, filename):
    torch.save(self.net.state_dict(), filename)

  def set(self, W):
    list(self.net.parameters())[0].data.copy_(W)

  def orthogonalityUpdate(self, beta):
    W = list(self.net.parameters())[0].data
    # update to keep W orthogonal
    W = (1+beta) * W
    W.addmm(-beta, torch.mm(W, W.t()), W)
    self.set(W)


class Discriminator(nn.Module):
  def __init__(self, args):
    super(Discriminator, self).__init__()

    print("* Discriminator model initialiation")
    # the discriminator - outputs log probability of input to be source or target
    self.net = nn.Sequential(
          nn.Dropout(args.discDropout),
          nn.Linear(args.dim, args.hidden, True),
          nn.LeakyReLU(),
          nn.Linear(args.hidden, 1),
          nn.Sigmoid()
        )
    for param in self.net.parameters():
      param.data.uniform_(-0.1,0.1)
    print(self.net)

  def forward(self, x):
    return self.net.forward(x)
