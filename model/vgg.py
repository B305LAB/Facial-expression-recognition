import torch
from torch import nn
import math
import torch.nn.functional as F
# import torch.utils.model_zoo as model_zoo

FEATURE_SIZE = 2048
#def conv_block(in_channel, out_channel):
    #layer = nn.Sequential(
     #   nn.BatchNorm2d(in_channel),
      #  nn.ReLU(),
      #  nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False)
   # )
   # return layer

#class dense_block(nn.Module):
   # def __init__(self, in_channel, growth_rate, num_layers):
      #  super(dense_block, self).__init__()
    #    block = []
     #   channel = in_channel
    #    for i in range(num_layers):
     #       block.append(conv_block(channel, growth_rate))
     #       channel += growth_rate
     #   self.net = nn.Sequential(*block)
  #  def forward(self, x):
   #     for layer in self.net:
   #         out = layer(x)
    #        x = torch.cat((out, x), dim=1)
    #    return x

#def transition(in_channel, out_channel):
 #   trans_layer = nn.Sequential(
  #      nn.BatchNorm2d(in_channel),
   #     nn.ReLU(),
 #       nn.Conv2d(in_channel, out_channel, 1),
  #      nn.AvgPool2d(2, 2)
  #  )
  #  return trans_layer

#class VGG(nn.Module):
 #   def __init__(self, num_classes=7, growth_rate=32, block_layers=[6, 12, 24, 16],pretrained=True):
  #      super(VGG, self).__init__()
  #      self.pretrained = pretrained
  #      self.block1 = nn.Sequential(
   #         nn.Conv2d(3, 64, 7, 2, 3),
  #          nn.BatchNorm2d(64),
   #         nn.ReLU(True),
    #        nn.MaxPool2d(3, 2, padding=1)
    #        )
    #    self.DB1 = self._make_dense_block(64, growth_rate,num=block_layers[0])
     #   self.TL1 = self._make_transition_layer(256)
    #    self.DB2 = self._make_dense_block(128, growth_rate, num=block_layers[1])
    #    self.TL2 = self._make_transition_layer(512)
    #    self.DB3 = self._make_dense_block(256, growth_rate, num=block_layers[2])
     #   self.TL3 = self._make_transition_layer(1024)
    #    self.DB4 = self._make_dense_block(512, growth_rate, num=block_layers[3])
    #    self.global_average = nn.Sequential(
    # #       nn.BatchNorm2d(1024),
        #      nn.ReLU(),
     ##       nn.AdaptiveAvgPool2d((1,1))
      #  )
        #self.layer_mid = nn.Sequential(
            #nn.Conv2d(512, 1024, kernel_size=7, padding=3),
            #nn.BatchNorm2d(1024),
            #nn.ReLU(inplace=True)


       # )
      #  self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        # self.classifier = nn.Linear(1024, num_classes)
       # for m in self.modules():
            # 判断m属于哪个类型
          #  if isinstance(m, nn.Conv2d):
            #    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            #    if m.bias is not None:
            #        m.bias.data.zero_()
         #   elif isinstance(m, nn.BatchNorm2d):
          #      nn.init.constant_(m.weight, 1)
           #     nn.init.constant_(m.bias, 0)

      #  if self.pretrained:
       #     self._load_pretrained()

   # def _load_pretrained(self):
    #    state_dict = torch.load( "/home/ubuntu/Downloads/facial_expression/CODE/self_atttention_for_ER/cache/densenet121-a639ec97 (1).pth")
    #    for key, value in state_dict.items():

      #     print("\nKey:"+key)
   # def forward(self, x):
      #  down1 = self.block1(x)
   #    down2 = self.DB1(down1)
      #  down4 = self.TL1(down2)
      #  down8 = self.DB2(down4)
      #  down16 = self.TL2(down8)
      #  down32 = self.DB3(down16)
       # down64 = self.TL3(down32)
       # down128 = self.DB4(down64)
       # down256 = self.global_average(down128)
        #x = x.view(x.shape[0], -1)
        #x = self.classifier(x)
       # l6 = torch.squeeze(self.gap(down256))

        #m6 = torch.squeeze(self.gap(self.layer_mid(down8)))

        #merge = torch.cat([l6, m6], dim=-1)

      #  return l6

  #  def _make_dense_block(self,channels, growth_rate, num):
    #    block = []
     #  block.append(dense_block(channels, growth_rate, num))
      #  channels += num * growth_rate

      #  return nn.Sequential(*block)
  #  def _make_transition_layer(self,channels):
      #  block = []
      #  block.append(transition(channels, channels // 2))
      #  return nn.Sequential(*block)
class VGG(nn.Module):

    def __init__(self, pretrained=True):
        super(VGG, self).__init__()
        self.pretrained = pretrained

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3,padding=1),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.GroupNorm(32, 512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.GroupNorm(32, 512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.GroupNorm(32, 512),
            nn.ReLU(inplace=True)

        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.GroupNorm(32, 512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.GroupNorm(32, 512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.GroupNorm(32, 512),
            nn.ReLU(inplace=True)

        )

        self.layer6 = nn.Sequential(
            #nn.Conv2d(512, 512,kernel_size=1,padding=1,dilation=5),
            #nn.GroupNorm(32, 512),
            #nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1,dilation=2),
            nn.GroupNorm(32, 1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.GroupNorm(32, 1024),
            nn.ReLU(inplace=True)
        )
        #self.se = nn.Sequential(
           # nn.AdaptiveAvgPool2d((1, 1)),
           # nn.Conv2d(256, 256 // 16, kernel_size=1),
           # nn.ReLU(),
            #nn.Conv2d(256 // 16, 256, kernel_size=1),
           # nn.Sigmoid()
       # )



        self.layer_mid = nn.Sequential(
            nn.Conv2d(256, 1024, kernel_size=7, padding=3),
            nn.GroupNorm(32, 1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=1, padding=1),
            nn.GroupNorm(32, 1024),
            nn.ReLU(inplace=True)

        )



        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        #权值参数初始化
        for m in self.modules():
            #判断m属于哪个类型
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if self.pretrained:
            self._load_pretrained()

    def _load_pretrained(self):
       # "Loading pretrained weights of conv layers from vgg16"
        state_dict = torch.load("/home/ubuntu/Downloads/facial_expression/CODE/self_atttention_for_ER/cache/vgg_vd_face_sfew_dag.pth")
        #for key,value in state_dict.items():
            #print("\nKey:"+key)


        self.layer1[0].weight.data.copy_(state_dict['conv1_1.weight'].detach())
        self.layer1[0].bias.data.copy_(state_dict['conv1_1.bias'].detach())
        self.layer1[3].weight.data.copy_(state_dict['conv1_2.weight'].detach())
        self.layer1[3].bias.data.copy_(state_dict['conv1_2.bias'].detach())

        self.layer2[0].weight.data.copy_(state_dict['conv2_1.weight'].detach())
        self.layer2[0].bias.data.copy_(state_dict['conv2_1.bias'].detach())
        self.layer2[3].weight.data.copy_(state_dict['conv2_2.weight'].detach())
        self.layer2[3].bias.data.copy_(state_dict['conv2_2.bias'].detach())

        self.layer3[0].weight.data.copy_(state_dict['conv3_1.weight'].detach())
        self.layer3[0].bias.data.copy_(state_dict['conv3_1.bias'].detach())
        self.layer3[3].weight.data.copy_(state_dict['conv3_2.weight'].detach())
        self.layer3[3].bias.data.copy_(state_dict['conv3_2.bias'].detach())
        self.layer3[6].weight.data.copy_(state_dict['conv3_3.weight'].detach())
        self.layer3[6].bias.data.copy_(state_dict['conv3_3.bias'].detach())


        self.layer4[0].weight.data.copy_(state_dict['conv4_1.weight'].detach())
        self.layer4[0].bias.data.copy_(state_dict['conv4_1.bias'].detach())
        self.layer4[3].weight.data.copy_(state_dict['conv4_2.weight'].detach())
        self.layer4[3].bias.data.copy_(state_dict['conv4_2.bias'].detach())
        self.layer4[6].weight.data.copy_(state_dict['conv4_3.weight'].detach())
        self.layer4[6].bias.data.copy_(state_dict['conv4_3.bias'].detach())


        self.layer5[0].weight.data.copy_(state_dict['conv5_1.weight'].detach())
        self.layer5[0].bias.data.copy_(state_dict['conv5_1.bias'].detach())
        self.layer5[3].weight.data.copy_(state_dict['conv5_2.weight'].detach())
        self.layer5[3].bias.data.copy_(state_dict['conv5_2.bias'].detach())
        self.layer5[6].weight.data.copy_(state_dict['conv5_3.weight'].detach())
        self.layer5[6].bias.data.copy_(state_dict['conv5_3.bias'].detach())


        #self.layer3[0].weight.data.copy_(state_dict['conv3_1.weight'].detach())
        #self.layer3[0].bias.data.copy_(state_dict['conv3_1.bias'].detach())



    def forward(self, x):
        down1 = self.layer1(x)

        down2 = self.maxpool(down1)
        down2 = self.layer2(down2)

        down4 = self.maxpool(down2)
        down4 = self.layer3(down4)

        #down4 = self.maxpool(down4)
       # down64 = self.se(down4)
       # down4 = down4*down64

        down8 = self.maxpool(down4)
        down8 = self.layer4(down8)

        down16 = self.maxpool(down8)
        down16 = self.layer5(down16)

        down32 = self.maxpool(down16)
        down32 = self.layer6(down32)







        #down32 = self.up_channel1(down32)
        l6 = torch.squeeze(self.gap(down32))

        m6 = torch.squeeze(self.gap(self.layer_mid(down4)))

        merge = torch.cat([l6, m6], dim=-1)

        return merge
class SelfAttention(nn.Module):

    def __init__(self, input_size=FEATURE_SIZE, output_size=FEATURE_SIZE):
        super(SelfAttention, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.K = nn.Linear(in_features=self.input_size, out_features=self.output_size, bias=False)
        self.Q = nn.Linear(in_features=self.input_size, out_features=self.output_size, bias=False)
        self.V = nn.Linear(in_features=self.input_size, out_features=self.output_size, bias=False)
        self.output_linear = nn.Linear(in_features=self.output_size, out_features=self.input_size, bias=False)

        self.drop50 = nn.Dropout(0.5)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)


    def forward(self, y):
        b = y.shape[0]
        n = y.shape[1]

        K = self.K(y)  # （n,m）->(n,?), ?=hidden size
        Q = 0.1 * self.Q(y) # 0.06
        V = self.V(y)

        logits = torch.matmul(Q, K.transpose(2, 1))

        # ignore itself
        eyes = torch.eye(n).byte()
        eyes = eyes.repeat(b, 1, 1)
        logits[eyes] = -float("Inf")

        attention_weights = nn.functional.softmax(logits, dim=1)
        weights = self.drop50(attention_weights)
        y = torch.matmul(V.transpose(2, 1), weights).transpose(2, 1)
        y = self.output_linear(y)

        return y, attention_weights





class ERNet(nn.Module):
    def __init__(self):
        super(ERNet, self).__init__()

        # self.VGG = VGG(True)


        self.SelfAttention = SelfAttention()
        self.TimeDistributed = TimeDistributed(VGG())
        self.drop50 = nn.Dropout(0.5)
        self.layer_norm = nn.LayerNorm(FEATURE_SIZE)
        self.relu = nn.ReLU()
        #self.encoder = nn.LSTM(input_size=(10,FEATURE_SIZE), hidden_size=1024,
                              # num_layers=1, bidirectional=None,
                              # dropout=0.5)
        self.linear_1 = nn.Linear(in_features=FEATURE_SIZE, out_features=FEATURE_SIZE, bias=False)
        self.linear_2 = nn.Linear(in_features=FEATURE_SIZE, out_features=7, bias=False)
        self.out = nn.Softmax()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, batch_long_x):
        batch_long_f = self.TimeDistributed(batch_long_x)



        y, attention_weights = self.SelfAttention(batch_long_f)
        y = self.layer_norm(y)
        y = y.sum(-2) / batch_long_x.size(1)
       # y = self.encoder(y)
        y = self.relu(y)
        y = self.drop50(y)
        # y = self.layer_norm(y)
        # y = self.linear_1(y)
        # y = self.relu(y)
        # y = self.drop50(y)
        # y = self.layer_norm(y)
        y = self.linear_2(y)
        y = self.out(y)

        return y, attention_weights

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        # if len(x.size()) <= 2:
        #     return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-3), x.size(-2), x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), x.size(1), -1)  # (samples, timesteps, output_size)
        else:
            # y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
            pass
        return y



