import torch
import torch.nn as nn
from torch.nn import functional as F

# 1. データの準備
text = """吾輩は猫である。名前はまだ無い。どこで生れたか頓（とん）と見当がつかぬ。
何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。
吾輩はここで始めて人間というものを見た。"""

# 文字の一覧を作成（語彙）
# 文字を辞書順にソート　
# ['\n', '。', 'あ', 'い', 'う', 'か', 'が', 'け', 'こ', 'し', 'じ', 'た', 'だ', 'つ', 'て', 'で', 'と', 'ど', 'ぬ', 'の', 'は', 'ま', 'め', 'も', 'る', 'れ', 'を', 'ん', 'ニ', 'ャ', 'ー', '事', '人', '何', '前', '名', '吾', '始', '当', '憶', '所', '暗', '泣', '無', '猫', '生', '薄', '見', '記', '輩', '間', '頓', '（', '）']
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# データをテンソルに変換
data = torch.tensor(encode(text), dtype=torch.long)


# 2. 非常にシンプルなモデルの定義（クラス名のスペースを修正：BigramLanguageModel）
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # 各文字の「次に来る文字の確率」を保持するテーブル
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        if targets is None:
            loss = None
        else:
            # PyTorchのクロスエントロピー誤差関数の期待する形状に変形
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            # 最後の文字の予測を取り出す
            logits = logits[:, -1, :]
            # 確率分布に変換
            probs = F.softmax(logits, dim=-1)
            # 次の1文字をサンプリング
            idx_next = torch.multinomial(probs, num_samples=1)
            # 文末に追加
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# モデルのインスタンス化
model = BigramLanguageModel(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2) # 学習率を少し上げました

# 3. 学習ループ
print("--- 学習開始前の生成結果 ---")
# 最初の入力として「0番目」の文字（あるいはダミー）を与える
context = torch.zeros((1, 1), dtype=torch.long)
print(decode(model.generate(context, max_new_tokens=20)[0].tolist()))

print("\n学習中...")
for steps in range(1001):
    # 全データを使って学習（ターゲットは1文字ずらしたもの）
    inputs = data[:-1].view(1, -1)
    targets = data[1:].view(1, -1)
    
    logits, loss = model(inputs, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    if steps % 200 == 0:
        print(f"step {steps}: loss {loss.item():.4f}")

print("\n--- 学習後の生成結果 ---")
print(decode(model.generate(context, max_new_tokens=30)[0].tolist()))