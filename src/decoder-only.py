import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import numpy as np
import matplotlib.pyplot as plt

batch_size = 64 
block_size = 256 
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 30
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 写出text的所有字符，并进行排序
chars = sorted(list(set(text)))
vocab_size = len(chars)
# 构建整数和字符的映射
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] 
decode = lambda l: ''.join([itos[i] for i in l]) 

# 按9：1的比例分配训练集和测试集
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
test_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size)
        self.query = nn.Linear(n_embd, head_size)
        self.value = nn.Linear(n_embd, head_size)
        # register_buffer适合用于保存模型的常量或中间状态
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        out = q @ k.transpose(-2,-1)
        out = out * k.shape[-1]**-0.5
        out = out.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        out = F.softmax(out, dim=-1)
        out = self.dropout(out)
        out = out @ v
        return out

class MultiHead(nn.Module):
    def  __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
    
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, num_heads):
        super().__init__()
        head_size = n_embd // num_heads
        self.mt = MultiHead(num_heads, head_size) #多头注意力
        self.ff = FeedForward(n_embd) #前馈神经网络
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.mt(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) 
        # self.position_embedding_table = nn.Embedding(block_size, n_embd) #自学习位置编码
        self.blocks = nn.Sequential(*[Block(n_embd, num_heads=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # 层归一化
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        # pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb # + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# 绘制损失曲线  
def draw_loss(train_loss, test_loss, fig_name):
    save_dir = ''
    
    save_path = os.path.join(save_dir, fig_name)
    
    # 创建x轴数据（epoch索引），长度与训练损失列表相同
    x = np.arange(len(train_loss))
    
    # 创建图形窗口，设置大小为12×5英寸
    plt.figure(figsize=(12, 5))
    
    # 绘制训练损失曲线
    plt.plot(x, train_loss, label="Train Loss", linewidth=1.5)
    
    # 绘制测试损失曲线
    plt.plot(x, test_loss, label="Test Loss", linewidth=1.5)
    
    # 设置x轴标签
    plt.xlabel("Epoch")
    
    # 设置y轴标签
    plt.ylabel("Loss")
    
    # 设置子图标题
    plt.title('Loss over Epochs')
    
    plt.legend()
    
    plt.tight_layout()
    
    plt.savefig(save_path)
    
    plt.show()

    return

model = GPTLanguageModel()
m = model.to(device)

# 使用Adamw优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

train_losses = []
test_losses = []

for iter in range(max_iters):

    losses = estimate_loss()
    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['test']:.4f}")
    train_losses.append(losses['train'])
    test_losses.append(losses['test'])
    
    xb, yb = get_batch('train')

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

fig_name = 'result4.jpg'
draw_loss(train_losses, test_losses, fig_name)

context = torch.zeros((1, 1), dtype=torch.long, device=device)
open('output4.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))                                                                                                                    