import torch 
from icecream  import ic
import torchvision.models as models 

# ic(dir(models))
mm = models.mobilenet_v2()
rn = models.resnet18()
ic(rn)


# # from tqdm import tqdm
# # from tqdm._utils import _term_move_up
# # import time

# # pbar = tqdm(range(5))
# # border = "="*50
# # clear_border = _term_move_up() + "\r" + " "*len(border) + "\r"
# # for i in pbar:
# #     pbar.write(clear_border + "Message %d" % i)
# #     pbar.write(border)
# #     pbar.update()
# #     time.sleep(1)


# from tqdm import trange
# from time import sleep
# t = trange(100, desc='Bar desc', leave=True)
# for i in t:
#     t.set_description("Bar desc (file %i)" % i)
#     t.refresh() # to show immediately the update
#     sleep(0.01)

# # ic(rn)


# # pytorch_total_params = sum(p.numel() for p in mm.parameters())
# # ic(pytorch_total_params)
pytorch_total_params = sum(p.numel() for p in rn.parameters())
ic(pytorch_total_params)
# pytorch_total_params = sum(p.numel() for p in alex.parameters())
# # ic(pytorch_total_params)

import torch 
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(10,5)
    
    def forward(self,x):
        return self.l1(x)

if __name__ == '__main__':
    k = Model()

    for name,i in k.parameters():
        i.register_backward_hook(lambda grad,_,o: print(grad))
    # print(dir(k))

    # for i in k.named_parameters():
    #     print(i)
    x = torch.randn((10,10))
    y = k(x)
    g = y.sum()
    g.backward()
    