# RL Hello World — 強化學習學習計劃

從零開始，用最小可執行的範例理解每個核心概念。
每個階段都有對應的程式碼，跑起來、改參數、看結果。

---

## 進度總覽

| # | 檔案 | 演算法 | 環境 | 狀態 |
|---|------|--------|------|------|
| 1 | `rl_helloworld.py` | Q-Learning | GridWorld (4×4) | ✅ 完成 |
| 2 | `dqn_cartpole.py` | DQN | CartPole-v1 | ✅ 完成 |
| 3 | `pg_cartpole.py` | Policy Gradient (REINFORCE) | CartPole-v1 | ✅ 完成 |
| 4 | `actor_critic.py` | Actor-Critic (A2C, MC) | CartPole-v1 | ✅ 完成 |
| 4b | `actor_critic_td.py` | Actor-Critic (A2C, TD) | CartPole-v1 | ✅ 完成 |
| 5 | `ppo_cartpole.py` | PPO (PPO-Clip) | CartPole-v1 | ✅ 完成 |
| 6 | `ppo_custom_env.py` | PPO | 自訂環境 | ⬜ 待做 |

---

## 階段說明

### ✅ 階段 1 — Q-Learning on GridWorld
**核心概念：** State、Action、Reward、Q-table、Bellman equation、ε-greedy

Q-table 直接存每個 (s, a) 的預期獎勵。
狀態是離散的格子座標，16 種狀態全部裝得進表格。

```
學到的東西：RL 的基本循環
  觀察狀態 → 選動作 → 得獎勵 → 更新 Q 值 → 重複
```

---

### ✅ 階段 2 — DQN on CartPole
**核心概念：** 函數近似、Experience Replay、Target Network

狀態變成 4 個連續浮點數，Q-table 裝不下。
解法：用神經網路取代 Q-table。Bellman 更新邏輯完全沒變。

```
新增技巧：
  Experience Replay  → 打破時間相關性，訓練更穩定
  Target Network     → 避免追著自己的尾巴跑
```

**Q-table vs DQN 的本質差異：**
只有一個：把 `Q[s][a]` 查表換成 `network(s)[a]` 計算。

> 這也是整個系列從 **numpy → PyTorch** 的切換點：階段 1 沒有神經網路，
> 用 numpy 最清楚；這裡開始有了網路，交給 PyTorch 的 autograd 與 Adam。

---

### ✅ 階段 3 — Policy Gradient on CartPole
**核心概念：** Policy-based vs Value-based、REINFORCE、Monte Carlo return

DQN 學的是「Q 值」，動作從 argmax 推出來（間接）。
Policy Gradient 直接學「動作機率」，網路輸出 softmax。

```
核心公式：
  ∇J = Σ ∇log π(a|s) × G_t

  G_t = 從這步開始到結尾的累積獎勵（Monte Carlo）
  → 做對了的動作，提高它的機率；做錯了的，降低
```

**為什麼要學這個：**
Actor-Critic、PPO 全部建立在這個基礎上。
RLHF（ChatGPT 的訓練方式）用的就是 PPO。

---

### ✅ 階段 4 — Actor-Critic (A2C，MC 版)
**核心概念：** Baseline、Advantage、Actor + Critic 雙網路

Policy Gradient 的問題：G_t 的方差很大，訓練不穩定。
解法：加一個 Critic 網路來估 V(s)，用 Advantage 取代原始 return。

```
Advantage = Q(s,a) - V(s)
          = 「這個動作比平均好多少」

Actor  → 學 policy π(a|s)，決定動作
Critic → 學 value V(s)，評估狀態好不好
```

---

### ✅ 階段 4b — Actor-Critic (A2C，TD 版)
**核心概念：** TD target、TD error = Advantage、bootstrap

和階段 4 唯一的差別：把 MC return 換成 TD target。
不用等整個 episode 跑完，每走一步就能算 advantage。

```
MC 版（階段 4）：G_t = 到 episode 結尾的累積折扣獎勵
  → 無 bias，但方差高，要等 episode 結束

TD 版（本檔）：δ_t = r_t + γ × V(s_{t+1}) - V(s_t)
  → 有 bias、方差低，每步即時可算
  → 這個 δ_t 就是 TD 版的 Advantage

⚠️ terminated（桿子倒）：V(s')=0；truncated（時間截斷）：V(s') 照常 bootstrap
```

這版是 PPO 的直接前身——階段 5 的 advantage 就沿用這裡的單步 TD δ。

---

### ✅ 階段 5 — PPO on CartPole
**核心概念：** Clipped Objective、ratio、同批多 epoch 複用

Actor-Critic 的問題：每次更新幅度不好控制，一步走太大就崩掉；
而且 on-policy 資料用一次就丟，樣本效率低。
PPO 用 clip 限制每次 policy 的更新幅度，於是同一批資料可以安全地
反覆更新好幾個 epoch。

```
PPO Objective：
  L = min( r_t × A_t,  clip(r_t, 1-ε, 1+ε) × A_t )

  r_t = π_new(a|s) / π_old(a|s)  （新舊 policy 的比值）
  → 比值偏離 1 太多就截斷，保守更新
```

**本檔定位（教科書核心版，嚴格最小 diff）：**
只放 PPO 真正的核心——`ratio + clip`（ε=0.2）與「同批整批複用 K 次」；
advantage 沿用階段 4b 的單步 TD δ。刻意**不放** GAE、entropy bonus、
共享網路、advantage 正規化，**也不切 minibatch**（minibatch 是正交的
SGD 工程技巧，AC 一樣能用，非 PPO 核心），讓 diff 精準隔離出
「PPO = A2C(TD) + clip」。為求乾淨對照，batch（4）、更新次數（250）、
actor 正規化（`.sum()/N`）、lr（actor 0.001 / critic 0.005）全部對齊
階段 4b。

```
相比 actor_critic_td.py 的唯一改動（只有這三點）：
  1. 更新前凍結 old_log_prob（π_old）與 advantage / TD target
  2. 目標  log π · A  →  min(ratio·A, clip(ratio)·A)，ratio 做 important sampling
  3. 外層加「整批複用 K 次」迴圈，重複用同一批資料（不切 minibatch）
```

> 細節：用 `/N`（軌跡數）正規化而非 `.mean()`，與整個系列一致——這讓
> PPO「第一個 epoch」的梯度精確等於 AC(TD)（ratio=1 時 ∇(ratio·A)=∇log π·A），
> PPO 因此是 AC(TD) 不打折的乾淨超集。

**為什麼以 PPO 收尾：**
PPO 長期是 RLHF 的標準引擎——在經典流程裡，reward model 訓練完之後，
就是用 PPO 拿它當分數來微調語言模型（InstructGPT / 早期 ChatGPT 就是這樣）。
它穩定、通用，是理解現代 LLM 對齊的基礎。

不過 2024 年起這塊已經分流出更新的方法，值得知道：
  - DPO：跳過獨立 reward model 與 PPO，直接用偏好資料優化，較簡單
  - GRPO：拿掉 critic，一個 prompt 生一組答案互相比較
          （DeepSeek R1 用它訓練推理模型，現為開源界主流）
  - 大趨勢：從「學出來的 reward model」轉向 verifiable rewards（程式自動驗對錯）

換句話說，PPO 不是 RL 的句點，而是看懂後續這些方法的起點。

---

### ⬜ 階段 6 — PPO on 自訂環境
**核心概念：** Gymnasium 介面、reward shaping、自己設計問題

用 `gymnasium` 包一個自己的問題（例如：簡單交易環境、迷宮、排程問題）。
把前面學的 PPO 套上去，體驗「定義問題」比「改演算法」更重要。

---

## 演算法家譜

```
強化學習
├── Value-based（學 Q 值）
│   ├── Q-Learning          ← 階段 1
│   └── DQN                 ← 階段 2
│
└── Policy-based（學 policy）
    ├── Policy Gradient / REINFORCE   ← 階段 3
    └── Actor-Critic
        ├── A2C (MC)         ← 階段 4
        ├── A2C (TD)         ← 階段 4b
        └── PPO              ← 階段 5、6
```

> **關於「嚴格最小 diff」**：policy-gradient 鏈（pg → AC(MC) → AC(TD) →
> PPO）刻意把 batch（4）、更新次數（250）、總 episodes（1000）、actor 正規化
> （`.sum()/N`）、gamma（0.99）、網路架構全部對齊，好讓階段間的 diff 只剩
> 「核心概念」。**唯一沒被隔離的變數是 learning rate**：每階段各自重調
> （pg 0.01 → AC(MC) 0.0005/0.001 → AC(TD) 與 PPO 0.001/0.005）。原因是 lr
> 與演算法本質耦合（return / advantage / TD δ 的天然尺度不同，TD 又特別吃
> critic 準度），屬「必要的重調」而非正交超參，故保留但明示。

---

## 核心概念速查

| 概念 | 白話 |
|------|------|
| State | Agent 目前看到的情況 |
| Action | Agent 能做的選擇 |
| Reward | 環境給的即時分數 |
| Policy π(a\|s) | 在狀態 s 下選動作 a 的機率 |
| Value V(s) | 從狀態 s 開始，預期能拿到多少總獎勵 |
| Q(s,a) | 在狀態 s 採取動作 a，預期能拿到多少總獎勵 |
| Advantage | 某動作比平均好多少：Q(s,a) - V(s) |
| Bellman | Q(s,a) = r + γ × max Q(s', a')，用遞迴定義 Q 值 |
| ε-greedy | 以 ε 機率探索，1-ε 機率利用已知最佳動作 |
| On-policy | 用當前 policy 產生的資料來訓練（PPO） |
| Off-policy | 可以用舊資料訓練（DQN） |

---

## 環境設定

依賴 `gymnasium`、`numpy`、`torch`。
階段 1 只用到 numpy（Q-table）；階段 2 起的神經網路用 PyTorch。

```bash
pip install -r requirements.txt
# 或手動： pip install "gymnasium[classic-control]" numpy torch
```

> CartPole 的網路很小，CPU 就跑得動，安裝 CPU 版 torch 即可：
> `pip install torch --index-url https://download.pytorch.org/whl/cpu`

執行範例：

```bash
python rl_helloworld.py     # 階段 1  Q-Learning
python dqn_cartpole.py      # 階段 2  DQN
python pg_cartpole.py       # 階段 3  Policy Gradient
python actor_critic.py      # 階段 4  A2C (MC)
python actor_critic_td.py   # 階段 4b A2C (TD)
python ppo_cartpole.py      # 階段 5  PPO
```
