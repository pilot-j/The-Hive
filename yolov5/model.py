import time
import torch
import torch.nn as nn
from torchvision.transforms import Resize
from torchvision.transforms import InterpolationMode
import config
from utils.utils import check_size, count_parameters

class Focus(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        
    def forward(self, x):  #during concat channels add
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        '''
        Official documentation defines a class Contract with param gain. 
        gain = a, downsamples height and width channel by factor a.
        self.contract= Contract(gain=2)
        //
        return self.conv(self.contract(x))
        '''
     
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
    c_ = int(c2 * e)  # hidden channels
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
class YOLOV5m(nn.Module):
    def __init__(self, c_out = b, nc=3, anchors=(),ch=(), inference=False):
        super().__init__()
        self.inference = inference
        self.backbone = nn.ModuleList()
        self.backbone += [
            Conv(3, b ,6, 2, 2),
            Conv(b, b*2, 3, 2, 1),
            C3(b*2, b*2, n=2),
            Conv(b*2, b*4, 3, 2, 1),
            C3(b*4, b*4, n=4),
            Conv(b*4, b*8, 3, 2, 1),
            C3(b*8, b*8, n=6),
            Conv(b*8, b*16, 3, 2, 1),
            C3(b*16, b*16, n=2),
            SPPF(b*16, b*16)
        ]
        self.neck = nn.ModuleList()
        self.neck += [
            CBL(b*16, b*8, 1, 1, 0),
            C3(b*16, b*8, n=2,shortcut=False, e=0.25),
            CBL(b*8, b*4, 1, 1, 0),
            C3(b*8, b*4, n=2, shortcut=False, e=0.25),
            CBL(b*4, b*4, 3, 2, 1),
            C3(b*8, b*8, n=2, shortcut=False),
            CBL(b*8, b*8, 3, 2, 1),
            C3(b*16, b*16, n=2, shortcut=False)
        ]
        def forward(self, x):
        assert x.shape[2] % 32 == 0 and x.shape[3] % 32 == 0, "Width and Height aren't divisible by 32!" #assertion error
        backbone_connection = [] #output from backbone, fed to neck
        neck_connection = [] #output from neck
        outputs = [] #fed to head blocks
        for idx, layer in enumerate(self.backbone):
            # takes the out of the 2nd and 3rd C3 block and stores it
            x = layer(x)
            if idx in [4, 6]:
                backbone_connection.append(x)

        for idx, layer in enumerate(self.neck):
            if idx in [0, 2]:
                x = layer(x)
                neck_connection.append(x)
                x = nn.Upsample(scale_factor=2, mode='nearest')(x)
                #x = Resize([x.shape[2]*2, x.shape[3]*2], interpolation=InterpolationMode.NEAREST)(x)
                x = torch.cat([x, backbone_connection.pop(-1)], dim=1)

            elif idx in [4, 6]:
                x = layer(x)
                x = torch.cat([x, neck_connection.pop(-1)], dim=1)

            elif isinstance(layer, C3) and idx > 2:
                x = layer(x)
                outputs.append(x)

            else:
                x = layer(x)
                
        return self.head(outputs)
    class HEADS(nn.Module):
    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(HEADS, self).__init__()
        self.nc = nc  # number of classes
        self.nl = len(anchors)  # number of detection layers
        self.naxs = len(anchors[0]) # number of anchors per scale
        self.stride = [8, 16, 32]

        # anchors are divided by the stride (anchors_for_head_1/8, anchors_for_head_1/16 etc.)
        anchors_ = torch.tensor(anchors).float().view(self.nl, -1, 2) / torch.tensor(self.stride).repeat(6, 1).T.reshape(3, 3, 2)
        self.register_buffer('anchors', anchors_) 

        self.out_convs = nn.ModuleList()
        for in_channels in ch:
            self.out_convs += [
                nn.Conv2d(in_channels=in_channels, out_channels=(5+self.nc) * self.naxs, kernel_size=1)
            ]

    def forward(self, x):
        for i in range(self.nl):
            x[i] = self.out_convs[i](x[i])
            bs, _, grid_y, grid_x = x[i].shape
            x[i] = x[i].view(bs, self.naxs, (5+self.nc), grid_y, grid_x).permute(0, 1, 3, 4, 2).contiguous()

        return x
    def cells_to_bboxes(predictions, anchors, strides):
    num_out_layers = len(predictions)
    grid = [torch.empty(0) for _ in range(num_out_layers)]  # initialize
    anchor_grid = [torch.empty(0) for _ in range(num_out_layers)]  # initialize   
    all_bboxes = []
    for i in range(num_out_layers):
        bs, naxs, ny, nx, _ = predictions[i].shape
        stride = strides[i]
        grid[i], anchor_grid[i] = make_grids(anchors, naxs, ny=ny, nx=nx, stride=stride, i=i)

        layer_prediction = predictions[i].sigmoid()
        
        obj = layer_prediction[..., 4:5]
        xy = (2 * (layer_prediction[..., 0:2]) + grid[i] - 0.5) * stride
        wh = ((2*layer_prediction[..., 2:4])**2) * anchor_grid[i]
        best_class = torch.argmax(layer_prediction[..., 5:], dim=-1).unsqueeze(-1)
        
        scale_bboxes = torch.cat((best_class, obj, xy, wh), dim=-1).reshape(bs, -1, 6)
        all_bboxes.append(scale_bboxes)

    return torch.cat(all_bboxes, dim=1)
 

def make_grids(anchors, naxs, stride, nx=20, ny=20, i=0):
    
    x_grid = torch.arange(nx)
    x_grid = x_grid.repeat(ny).reshape(ny, nx)
    y_grid = torch.arange(ny).unsqueeze(0)
    y_grid = y_grid.T.repeat(1, nx).reshape(ny, nx)
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)
    xy_grid = xy_grid.expand(1, naxs, ny, nx, 2)
    
    anchor_grid = (anchors[i]*stride).reshape((1, naxs, 1, 1, 2)).expand(1, naxs, ny, nx, 2)

    return xy_grid, anchor_grid
