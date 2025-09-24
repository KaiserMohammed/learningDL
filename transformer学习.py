#%%
import torch
import torch.nn as nn
#%%
class PositionalEncoding(nn.Module):#继承nn
    # Sine-cosine positional coding
    def __init__(self, emb_dim, max_len, freq=10000.0):#三个参数-嵌入维度emb——dim，最大长度，三角函数频率
        super(PositionalEncoding, self).__init__()#使用父类nn的初始化方法
        assert emb_dim > 0 and max_len > 0, 'emb_dim and max_len must be positive'#断言检查，看嵌入维度和最大长度是否都是正数
        self.emb_dim = emb_dim
        self.max_len = max_len#保存为类的实例变量
        self.pe = torch.zeros(max_len, emb_dim)#创建一个0向量，用于保存后续算出来的位置

        pos = torch.arange(0, max_len).unsqueeze(1)#创建一个从0到max_len-1的整数序列，并通过unsqueeze(1)增加一个维度，使形状变为[max_len, 1]，表示各个位置的索引
        # pos: [max_len, 1]
        div = torch.pow(freq, torch.arange(0, emb_dim, 2) / emb_dim)#计算位置编码的除数项，先生成步长为2的序列，再除嵌入维度使其归一化，通过torch.pow计算freq的上述次方，得到形状为[ceil(emb_dim/2)]的张量
        # div: [ceil(emb_dim / 2)]
        self.pe[:, 0::2] = torch.sin(pos / div)#偶数位置算正弦填入
        # torch.sin(pos / div): [max_len, ceil(emb_dim / 2)]
        self.pe[:, 1::2] = torch.cos(pos / (div if emb_dim % 2 == 0 else div[:-1]))#奇数位置算余弦填入
        # torch.cos(pos / div): [max_len, floor(emb_dim / 2)]

    def forward(self, x, len=None):
        if len is None:
            len = x.size(-2)#如果未指定len，则从输入x的形状中获取序列长度（假设x的倒数第二个维度是序列长度）。
        print(self.pe[:len, :])
        return x + self.pe[:len, :]#将输入x与前len个位置的编码相加，返回结果，实现了将位置信息注入到输入嵌入中的功能
#%%
class LearnablePositionalEncoding(nn.Module):
    # Learnable positional encoding
    def __init__(self, emb_dim, len):#输入嵌入维度和序列长度
        super(LearnablePositionalEncoding, self).__init__()
        assert emb_dim > 0 and len > 0, 'emb_dim and len must be positive'
        self.emb_dim = emb_dim
        self.len = len
        self.pe = nn.Parameter(torch.zeros(len, emb_dim))#torch.zeros(len, emb_dim)生成一个形状为[len, emb_dim]的零张量，初始值为 0。nn.Parameter()将该张量包装为 PyTorch 的可学习参数，意味着在训练过程中，这个矩阵的值会通过反向传播不断更新，以学习到最适合任务的位置信息

    def forward(self, x):
        return x + self.pe[:x.size(-2), :]#截取与输入序列长度匹配的位置编码部分（避免位置编码长度超过输入序列长度）
#不通过手工设计的正弦余弦函数来表示位置，而是让模型在训练过程中自动学习适合当前任务的位置特征。这种方式的灵活性更高，在某些任务上可能比固定的正弦 - 余弦编码表现更好，但只能处理预定义的最大长度
#%%
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, dim_qk=None, dim_v=None, num_heads=1, dropout=0.):#QK对应，维度必须相同，默认一头注意力不丢弃
        super(MultiHeadAttention, self).__init__()

        dim_qk = dim if dim_qk is None else dim_qk
        dim_v = dim if dim_v is None else dim_v

        assert dim % num_heads == 0 and dim_v % num_heads == 0 and dim_qk % num_heads == 0, 'dim must be divisible by num_heads'#注意力头数必须要能整除维度

        self.dim = dim
        self.dim_qk = dim_qk
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        self.w_q = nn.Linear(dim, dim_qk)#线性变化层的定义，用于映射
        self.w_k = nn.Linear(dim, dim_qk)
        self.w_v = nn.Linear(dim, dim_v)

    def forward(self, q, k, v, mask=None):
        # q: [B, len_q, D]
        # k: [B, len_kv, D]
        # v: [B, len_kv, D]
        assert q.ndim == k.ndim == v.ndim == 3, 'input must be 3-dimensional'#.ndim是torch方法，用于知道一个张量的维度，这里是在检查

        len_q, len_k, len_v = q.size(1), k.size(1), v.size(1)
        assert q.size(-1) == k.size(-1) == v.size(-1) == self.dim, 'dimension mismatch'
        assert len_k == len_v, 'len_k and len_v must be equal'
        len_kv = len_v#获取各序列的长度，检查特征维度是否一致，以及 k 和 v 的序列长度是否相等，并将 k 和 v 的序列长度统一记为len_kv

        q = self.w_q(q).view(-1, len_q, self.num_heads, self.dim_qk // self.num_heads)#对 q、k、v 进行线性变换后重塑张量，将特征维度拆分为num_heads个注意力头，每个头的维度为总维度除以头数
        k = self.w_k(k).view(-1, len_kv, self.num_heads, self.dim_qk // self.num_heads)#-1表示该维度自动，后续为长度，头数和每头维度
        v = self.w_v(v).view(-1, len_kv, self.num_heads, self.dim_v // self.num_heads)
        # q: [B, len_q, num_heads, dim_qk//num_heads]
        # k: [B, len_kv, num_heads, dim_qk//num_heads]
        # v: [B, len_kv, num_heads, dim_v//num_heads]
        # The following 'dim_(qk)//num_heads' is writen as d_(qk)

        q = q.transpose(1, 2)#交换维度，将注意力头的维度提前，便于后续计算。先拆在换不涉及内存移动，高效率
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # q: [B, num_heads, len_q, d_qk]
        # k: [B, num_heads, len_kv, d_qk]
        # v: [B, num_heads, len_kv, d_v]

        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.dim_qk ** 0.5)#注意力分数计算，其中.transpose为torch内置维度调换函数，实现转置
        # attn: [B, num_heads, len_q, len_kv]

        if mask is not None:
            attn = attn.transpose(0, 1).masked_fill(mask, float('-1e20')).transpose(0, 1)#如果提供了掩码，则对注意力分数应用掩码：将不需要关注的位置设置为一个极小值（接近负无穷）
            #这样在后续 softmax 计算时，这些位置的注意力权重会接近 0
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)#对注意力分数应用 softmax 得到注意力权重，并进行 dropout 操作（随机部分注意力权重置为零）防止过拟合。

        output = torch.matmul(attn, v)#将注意力权重与 v 相乘，得到每个位置的加权求和结果。
        # output: [B, num_heads, len_q, d_v]
        output = output.transpose(1, 2)#交换维度将序列长度提前
        # output: [B, len_q, num_heads, d_v]
        output = output.contiguous().view(-1, len_q, self.dim_v)#多头合并
        # output: [B, len_q, num_heads * d_v] = [B, len_q, dim_v]
        return output
#%%
def attn_mask(len):
    """
    :param len: length of sequence
    :return: mask tensor, False for not replaced, True for replaced as -inf
    e.g. attn_mask(3) =
        tensor([[[False,  True,  True],
                 [False, False,  True],
                 [False, False, False]]])
    生成一个上三角掩码，用于实现自注意力中的 "未来信息屏蔽"
    使用torch.triu生成上三角矩阵，对角线以上的元素设为 1（True）
    结果是一个形状为[1, len, len]的张量，上三角部分为 True（表示需要被替换为 - inf）
    应用场景：在解码器中防止模型关注未来的序列元素
    """
    mask = torch.triu(torch.ones(len, len, dtype=torch.bool), 1)
    return mask


def padding_mask(pad_q, pad_k):
    """
    :param pad_q: pad label of query (0 is padding, 1 is not padding), [B, len_q]
    :param pad_k: pad label of key (0 is padding, 1 is not padding), [B, len_k]
    :return: mask tensor, False for not replaced, True for replaced as -inf
    e.g. pad_q = tensor([[1, 1, 0]], [1, 0, 1])
        padding_mask(pad_q, pad_q) =
        tensor([[[False, False,  True],
                 [False, False,  True],
                 [ True,  True,  True]],
                [[False,  True, False],
                 [ True,  True,  True],
                 [False,  True, False]]])
                 输入参数：
    pad_q：查询序列的填充标记，形状为 [B, len_q]，0 表示填充，1 表示有效内容
    pad_k：键序列的填充标记，形状为 [B, len_k]，格式同上
    核心逻辑：
    通过unsqueeze操作扩展维度，将pad_q变为 [B, len_q, 1]，pad_k变为 [B, 1, len_k]
    使用矩阵乘法（*）得到一个 [B, len_q, len_k] 的张量，其中值为 True 表示对应位置都是有效内容
    取反（~）后，True 表示需要被掩码的位置（填充位置或与填充位置相关的注意力）
    输出：
    形状为 [B, len_q, len_k] 的掩码张量
    True 表示该位置在注意力计算中需要被替换为负无穷（-inf）
    False 表示该位置保持原样，参与正常计算
    例如当查询或键的对应位置有一个是填充（0）时，掩码结果就为 True，在注意力计算时会被忽略，这样就避免了填充部分对注意力分数的干扰。
    """
    assert pad_q.ndim == pad_k.ndim == 2, 'pad_q and pad_k must be 2-dimensional'
    assert pad_q.size(0) == pad_k.size(0), 'batch size mismatch'
    mask = pad_q.bool().unsqueeze(2) * pad_k.bool().unsqueeze(1)
    ''' pad_q.bool() 和 pad_k.bool() 将张量转换为布尔类型（True/False），通常填充位置为 True，有效内容为 False
    unsqueeze(2) 给 pad_q 增加一个第 2 维，形状从 [B, len_q] 变为 [B, len_q, 1]
    unsqueeze(1) 给 pad_k 增加一个第 1 维，形状从 [B, len_k] 变为 [B, 1, len_k]
    * 在这里是广播（broadcast）乘法，结果形状为 [B, len_q, len_k]，其中每个位置 (i,j) 为 True 表示该位置是填充区域需要被屏蔽
    作用：生成一个注意力掩码矩阵，标记出所有需要被屏蔽的（q, k）位置对 '''
    mask = ~mask#逻辑非，反转
    # mask: [B, len_q, len_k]
    return mask
#%%
class Feedforward(nn.Module):
    def __init__(self, dim, hidden_dim=2048, dropout=0., activate=nn.ReLU()):#隐藏层维度需要为原维度的2-4倍
        super(Feedforward, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(dim, hidden_dim)#升维层
        self.fc2 = nn.Linear(hidden_dim, dim)#降维层
        self.act = activate#使用输入的激活函数，否则RELU

    def forward(self, x):
        x = self.act(self.fc1(x))#升维
        x = self.dropout(x)#丢弃防止过拟合
        x = self.fc2(x)#降维
        return x
#%%
class EncoderLayer(nn.Module):
    def __init__(self, dim, dim_qk=None, num_heads=1, dropout=0., pre_norm=False):#pre_norm：布尔值，控制归一化操作的位置（是在子层前还是子层后）
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(dim, dim_qk=dim_qk, num_heads=num_heads, dropout=dropout)#实例化多头注意力，编码器中不需要掩码
        self.ffn = Feedforward(dim, dim * 4, dropout)
        self.pre_norm = pre_norm
        self.norm1 = nn.LayerNorm(dim)#用于自注意力的归一层
        self.norm2 = nn.LayerNorm(dim)#用于前馈神经网络的归一层

    def forward(self, x, mask=None):
        if self.pre_norm:#在子层前归一化
            res1 = self.norm1(x)
            x = x + self.attn(res1, res1, res1, mask)#残差链接
            res2 = self.norm2(x)
            x = x + self.ffn(res2)#前馈神经网络也要有一个残差链接
        else:#在子层后归一化
            x = self.attn(x, x, x, mask) + x
            x = self.norm1(x)
            x = self.ffn(x) + x
            x = self.norm2(x)
        return x
#%%
class Encoder(nn.Module):
    def __init__(self, dim, dim_qk=None, num_heads=1, num_layers=1, dropout=0., pre_norm=False):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(dim, dim_qk, num_heads, dropout, pre_norm) for _ in range(num_layers)])#通过列表推导式创建num_layers个相同配置的EncoderLayer实例，堆叠成编码器的深层结构。所有编码器层共享相同的参数配置（维度、头数、dropout 等），这是 Transformer 的标准设计。

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)#前一个编码器层的输出作为下一个编码器层的输入，实现特征的逐层传递和深度编码。
        return x
#%%
class DecoderLayer(nn.Module):
    def __init__(self, dim, dim_qk=None, num_heads=1, dropout=0., pre_norm=False):
        super(DecoderLayer, self).__init__()
        self.attn1 = MultiHeadAttention(dim, dim_qk=dim_qk, num_heads=num_heads, dropout=dropout)
        self.attn2 = MultiHeadAttention(dim, dim_qk=dim_qk, num_heads=num_heads, dropout=dropout)#两次多头注意力机制，第二次用于引入编码器
        self.ffn = Feedforward(dim, dim * 4, dropout)
        self.pre_norm = pre_norm
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, enc, self_mask=None, pad_mask=None):
        if self.pre_norm:
            res1 = self.norm1(x)
            x = x + self.attn1(res1, res1, res1, self_mask)#自注意力残差链接
            res2 = self.norm2(x)
            x = x + self.attn2(res2, enc, enc, pad_mask)#交叉注意力残差链接
            res3 = self.norm3(x)
            x = x + self.ffn(res3)
        else:
            x = self.attn1(x, x, x, self_mask) + x
            x = self.norm1(x)
            x = self.attn2(x, enc, enc, pad_mask) + x
            x = self.norm2(x)
            x = self.ffn(x) + x
            x = self.norm3(x)
        return x
#%%
class Decoder(nn.Module):
    def __init__(self, dim, dim_qk=None, num_heads=1, num_layers=1, dropout=0., pre_norm=False):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(dim, dim_qk, num_heads, dropout, pre_norm) for _ in range(num_layers)])

    def forward(self, x, enc, self_mask=None, pad_mask=None):#输入张量，输出张量，自注意力掩码，交叉注意力掩码
        for layer in self.layers:
            x = layer(x, enc, self_mask, pad_mask)
        return x
#%%

class Transformer(nn.Module):
    def __init__(self, dim, vocabulary, num_heads=1, num_layers=1, dropout=0., learnable_pos=False, pre_norm=False):
        super(Transformer, self).__init__()
        self.dim = dim
        self.vocabulary = vocabulary#词汇表大小
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.learnable_pos = learnable_pos#是否用可学习的位置编码
        self.pre_norm = pre_norm#是否在子层前归一化

        self.embedding = nn.Embedding(vocabulary, dim)
        self.pos_enc = LearnablePositionalEncoding(dim, 100) if learnable_pos else PositionalEncoding(dim, 100)
        self.encoder = Encoder(dim, dim // num_heads, num_heads, num_layers, dropout, pre_norm)
        self.decoder = Decoder(dim, dim // num_heads, num_heads, num_layers, dropout, pre_norm)
        self.linear = nn.Linear(dim, vocabulary)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, pad_mask=None):
        # src.shape: torch.Size([2, 10])
        src = self.embedding(src)#向量化
        # src.shape: torch.Size([2, 10, 512])
        src = self.pos_enc(src)#添加位置信息
        # src.shape: torch.Size([2, 10, 512])
        src = self.encoder(src, src_mask)#编码
        # src.shape: torch.Size([2, 10, 512])

        # tgt.shape: torch.Size([2, 8])
        tgt = self.embedding(tgt)#目标序列向量化
        # tgt.shape: torch.Size([2, 8, 512])
        tgt = self.pos_enc(tgt)#位置信息
        # tgt.shape: torch.Size([2, 8, 512])
        tgt = self.decoder(tgt, src, tgt_mask, pad_mask)#训练解码器
        # tgt.shape: torch.Size([2, 8, 512])

        output = self.linear(tgt)## 通过线性层将解码器输出转换为词汇表大小的概率分布
        print(output.shape)
        output_prob = torch.softmax(output, dim=-1)
        print(output_prob)
        return output_prob
        # output.shape: torch.Size([2, 8, 10000])
        return output

    def get_mask(self, tgt, src_pad=None):
        # Under normal circumstances, tgt_pad will perform mask processing when calculating loss, and it isn't necessarily in decoder
        if src_pad is not None:#生成填充掩码
            src_mask = padding_mask(src_pad, src_pad)
        else:
            src_mask = None
        tgt_mask = attn_mask(tgt.size(1))
        if src_pad is not None:#生成自注意力掩码
            pad_mask = padding_mask(torch.zeros_like(tgt), src_pad)
        else:
            pad_mask = None
        # src_mask: [B, len_src, len_src]
        # tgt_mask: [len_tgt, len_tgt]
        # pad_mask: [B, len_tgt, len_src]
        return src_mask, tgt_mask, pad_mask
#%%
if __name__ == '__main__':
    model = Transformer(dim=512, vocabulary=10000, num_heads=8, num_layers=6, dropout=0.1, learnable_pos=False,
                        pre_norm=True)
    src = torch.randint(0, 10000, (2, 10))  # torch.Size([2, 10]) # 生成随机源序列：形状为[2, 10]，值在0到10000之间（模拟2个样本，每个样本长度为10的序列）
    tgt = torch.randint(0, 10000, (2, 8))  # torch.Size([2, 8]) # 生成随机源序列：形状为[2, 10]，值在0到10000之间（模拟2个样本，每个样本长度为8序列）
    src_pad = torch.randint(0, 2, (2, 10))  # torch.Size([2, 10])
    src_mask, tgt_mask, pad_mask = model.get_mask(tgt, src_pad)#填充掩码
    model(src, tgt, src_mask, tgt_mask, pad_mask)
    # output.shape: torch.Size([2, 8, 10000])

#%%
