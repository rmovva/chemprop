import torch
import torch.nn as nn
import numpy as np
from hyperopt import pyll, hp

n_samples = 20

space = hp.qloguniform('x', low=np.log(10), high=np.log(100), q=10)
evaluated = [pyll.stochastic.sample(space) for _ in range(n_samples)]
# Output: [0.04645754, 0.0083128 , 0.04931957, 0.09468335, 0.00660693,
#          0.00282584, 0.01877195, 0.02958924, 0.00568617, 0.00102252]

q = 0.005
qevaluated = np.round(np.array(evaluated)/q) * q
print(qevaluated)

# input = torch.randn(2, 3, 4)
# >>> m = nn.LayerNorm(input.size()[1:])
# >>> m = nn.LayerNorm(input.size()[1:], elementwise_affine=False)
# >>> m = nn.LayerNorm([10, 10])
# m = nn.LayerNorm(4, 3)
# output = m(input)
# print(output)
# W_att = nn.Linear(4, 1)
# arr = torch.ones((2,3))
# arr = arr.unsqueeze(dim=2).repeat(1,1,4)
# arr = arr.sum(dim=2)
# print(arr)
# ans = W_att(arr)
# print(W_att, arr, ans)