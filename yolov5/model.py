import time
import torch
import torch.nn as nn
from torchvision.transforms import Resize
from torchvision.transforms import InterpolationMode
import config
from utils.utils import check_size, count_parameters

class Conv(nn.Module):
    default_act = nn.SiLU()
    def __init__(self, c1, c2, k =1, s=1, p='valid' , g=1 , d=1, act= True):
      super().__init__()
      self.conv=nn.Conv2d(c1 ,c2 ,kernel_size=1, stride=s, padding=p, groups=g, dilation =d, bias=False)
      self.bn = nn.BatchNorm2d(c2)
      nn.act=default_act if act is True else act if isinstance(act,nn.Module) else nn.Identity()
      
      def forward(self, x):
        return self.act(self.bn(self.conv(x)))
      def forward_plain(self,x):
        return self.act(self.conv(x))
        
 class Bottleneck(nn.Module):
  def __init__(self,c1,c2,skip=True,g=1, e=0.5):
    super().__init__()
    c_=int(c2*e)
    #c_=c2//w
    self.cv1=Conv(c1,c_,1,1)
    self.cv2=Conv(c_,c2,3,1, g=g) #why have we used g param here?

    def forward (self, x):
      return x + self.cv2(self.cv1(x)) if skip else self.cv2(self.cv1(x))
      
class C3(nn.Module):
  def __init__(self,c1,c2, n=1, shortcut=True, g=1, e=0.5):
    super().__init__()
    self.cv1 = Conv(c1,c_,1,1)
    self.cv2 = Conv(c_, c_,1, 1)
    self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
    self.cv3= Conv(2*c_, c2, 1, 1)

    def forward(self,x):
      return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)),1))
      
class SPPF(nn.Module):
  # equivalent to SPP(k=(5, 9, 13))
  def __init__(self, c1, c2, k=5):
    super().__init__()
    c_ = c1 // 2  # hidden channels
    self.cv1 = Conv(c1, c_, 1, 1)
    self.cv2 = Conv(c_ * 4, c2, 1, 1)
    self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
      x = self.cv1(x)
      with warnings.catch_warnings():
        warnings.simplefilter('ignore') 
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))
