import torch
from torch import nn
from ptflops import get_model_complexity_info


DEVICE = torch.device('cuda', 0)


net = nn.Sequential(
    nn.Linear(784, 128),
    nn.Tanh(),
    nn.Linear(128, 128),
    nn.Tanh(),
    nn.Linear(128, 10),
    nn.Softmax()
)

with torch.cuda.device(0):
  macs, params = get_model_complexity_info(net, (784,), as_strings=False,
                                           print_per_layer_stat=True, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))

xs = torch.zeros((128, 784)).to(DEVICE)
ys = torch.zeros((128, 1)).to(DEVICE)
net.to(DEVICE)

while True:
    net(xs)
    