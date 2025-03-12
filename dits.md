准备工作

```python
#clone
git clone https://ghfast.top/https://github.com/smile2game/mnist-dits.git
#免密
git config --global credential.helper store


```



# 3.10 手撕模型

## 基础知识：

### nn

1. 写torch类时，forward和\_\_init\_\_()应该是同时进行,

   1. 写forward时候，要用到的nn.Conv/Linear/Parameter/ReLU/Sequential/LayerNorm需要在 \_\_init\_\_种进行定义，

   2. 考虑好各个维度的分配

```python
  class DiT(nn.Module):
    def __init__():
    def forward():
```



* torch.exp/rand/arrange/view/expand等操作

  1. torch操作都是对整个张量直接操作

  2. view用于重整，expand会复制扩展

  3. rand和arrange用于生成



* nn.Conv2d/Embedding/Linear/LayerNorm/

  1. Conv2d可以用于patchify，(N,channel,H,W) >> (N,channel \* patch\_size \*\*2, row,col )

  2. Embedding用于将离散的类别标签（或索引）映射为连续的向量表示。映射**可学习的**，嵌入向量会在训练过程中通过反向传播进行优化。本质是 **查找表**

  3. Linear 全连接层&#x20;

     1. Linear是将 连续张量 >> 连续张量，用于特征映射与全连接

     2. Embedding是将 离散数值点 >> 连续张量，用于词汇或类别嵌入&#x20;

  4. LayerNorm





* nn的级别划分

  1. "层"（Layer）是nn的基本单元，负责对输入数据进行某种形式的变换（如线性变换、非线性激活等），称作成是因为层层叠叠

     1. Paramater 能够自定义层

  2. "块"(Module) / "块列表"(ModuleList) / 顺序容器(Sequential)

     1. 实际上Module是能嵌套同级的

  3. "模型"(model)

     1. model实际上也是Module







* nn.ModuleList / Sequential 有什么区别？

  1. ModuleList更灵活，动态删减且自定义forward

  2. Sequential就是很简单固定的顺序执行，且都是预先设定好的



* nn.Module到底继承了什么？

  1. 前向传播 forward()

  2. 参数管理nn.Parameter

  3. 设备管理.to(device)

  4. 子模块管理 children()/modules()

  5. 状态管理 state\_dict

  6. 钩子函数 register\_forward\_hook





### 张量乘法：

```python
half_emb_t=half_emb*t # (2,8)*(2,1)

#广播机制： 
1. 从左到右比较形状兼容： 相同或其中一个为1
2. 广播
3. 逐元素相乘


```









## train流程

```python
from config import *
#数据
dataset = MNIST()
BATCH_SIZE = 2000
dataloader = DataLoader(dataset,batch_size = BATCH_SIZE,shuffle = True, num_workers = 10,persistent_workers = True)

#模型
model =  DiT(img_size = 28,patch_size = 4,channel = 1,emb_size = 64,label_num = 10, dit)
model.train() #切换模式
#优化器 
optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3)
loss_fn = nn.L1Loss()

#训练
EPOCH = 500
iter_cnt = 0
for epoch in range(EPOCH):
    for imgs,label in dataloader:
        #数据预处理
        x = imgs * 2 - 1 #像素变换 [0,1] >> [-1,1]，符合噪音的高斯分布
        t = torch.randint(0,T,((imgs.size(0)),)) #每张图片随机的 t 时刻
        y = labels 
        
        #加噪与预测 
        x,noise = forward_add_noise(x,t)
        pred_noise = model(x.to(DEVICE), y.to(DEVICE),t.to(DEVICE))
        
        #梯度降
        loss = loss_fn(fred_noise,noise.to(DEVICE))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if iter_cnt % 1000 == 0:
            torch.save(model.state_dict(),f"model_{iter_cnt}.pth")
        iter_cnt +=1 

```



## 数据集&#x20;

\_\_init\_\_

```python
from torch.util.data import Dataset 
from torchvision.transforms.v2 import PILToTensor,Compose

class MNIST(Dataset)
    def __init__(self,is_train=True):
        super().__init__()
        self.ds = torchvision.datasets.MNIST('./MNIST/',train = is_train,download = True)
        self.img_convert = Compose([
            PILToTensor(), #
        ])
```



## 模型 构建

### DiT

```python
class DiT(nn.Module):
    def __init__(self,img_size,patch_size,channel,emb_size, label_num,dit_num, head):
        supuer().__init__()
        
        self.patch_size = patch_size
        self.patch_count = img_size // patch_size
        self.channel = channel
        
        #patchify
        self.conv = nn.Conv2d(in_channel= channel, out_channel= channel*patch_size**2,kernel_size = patch_size,stride =patch_size ) #patch在channel通道展平，后面是 行 列 坐标 
        self.patch_emb = nn.Linear(in_channel =  channel*patch_size**2, out_channel = emb_size)
        self.pos_emb = nn.Parameter(torch.rand(1,self.patch_count**2, emb)) #每个patch确保有一个pos_emb 
        
        #time_emb
        
        
        #label_emb
        
        #DiT Block
        
        
        #layer_norm
        
        #linear_back
    def forward():
        
        
```

### time\_emb

对每个时间步，生成sin-cos拼成的0-dim\_size = emb\_size的张量。

如果是一个批次，则得到 emb.shape = (N,emb\_size)

```python
class TimeEmbedding(nn.Module):
    def __init__(self,emb_size):
        super().__init__()
        self.half_emb_size=emb_size//2
        half_emb=torch.exp(torch.arange(self.half_emb_size)*(-1*math.log(10000)/(self.half_emb_size-1))) #构建衰减序列 e^([1,2,...] x -ln10000/(8-1)) >> shape = (8,)
        self.register_buffer('half_emb',half_emb)

    def forward(self,t):
        t=t.view(t.size(0),1) #t.shape = (N,1)
        half_emb=self.half_emb.unsqueeze(0).expand(t.size(0),self.half_emb_size) #half_emb.shape = (N,emb)
        half_emb_t=half_emb*t
        embs_t=torch.cat((half_emb_t.sin(),half_emb_t.cos()),dim=-1)
        return embs_t
```

###

### DiT-Block

```python
class DiTBlock(nn.Module):
    def __init__(emb_size,head): #创建时候的输入
        #condition
        self.gamma1 = nn.Linear(emb_size,emb_size)
        #layernorm
        self.ln1 = nn.LayerNorm(emb_size)
        
        #Attn
        wq/k/v = nn.Linear(emb_size, nhead*emb_size)
        wl = nn.Linear(nhead*emb_size , emb_size)
        
        #FFD
        self.ff  = nn.Sequential(
                    nn.Linear(emb_size,4*emb_szie),
                    nn.ReLU(),
                    nn.Linear(emb_size*8,emb_size)
                )
    def forward(x,cond): #调用时候的输入
        
        
```









## 加噪

forward\_add\_noise

```python
#构建噪声 占比系数 
betas = torch.linspace(0.0001.0.02,T) #噪声方差，前期小后期大 (T,）
alphas = 1-betas #保留原始信号的 比例 
alphas_cumprod = torch.cumprod(alphas,dim = -1) #alpha_t累乘 (T,)
alphas_cumprod_prev = torch.cat(torch.tensor([1.0]),alphas_cumprod[:-1]） ,dim =-1) #alpha_t-1累乘 (T,),整体后移一位，前面补个1。记作前一个时间步的累积

variance = (1- alphas)* (1-alphas_cumprod_prev)/(1-alphas_cumprod) #(T,) 后验方差 

def forward_add_noise(x,t):
    noise = torch.randn_like(x) #高斯噪声 
    batch_alphas_cumprod = alphas_cumprod[t].view(x.size(0),1,1,1) #构建噪声占比 
    x = torch.sqrt(batch_alphas_cumprod) * x +  torch.sqrt(1- batch_alphas_cumprod)* noise 
    return x,noise
    
```









# 3.11 手撕框架&#x20;

## 手撕ddp



## Deepspeed训练&#x20;



## Megatron训练&#x20;



## Accelerate训练



# 3.12手撕 并行&#x20;

## Odysseus

