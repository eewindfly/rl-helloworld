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
| 4 | `actor_critic.py` | Actor-Critic (A2C) | CartPole-v1 | ⬜ 待做 |
| 5 | `ppo_cartpole.py` | PPO | CartPole-v1 | ⬜ 待做 |
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

---

### ⬜ 階段 3 — Policy Gradient on CartPole
**核心概念：** Policy-based vs Value-based、REINFORCE、Monte Carlo return

DQN 學的是「Q 值」，動作從 argmax 推出來（間接）。
Policy Gradient 直接學「動作機率」，網路輸出 softmax。

```
核心公式：
  ∇J = Σ log π(a|s) × G_t

  G_t = 從這步開始到結尾的累積獎勵（Monte Carlo）
  → 做對了的動作，提高它的機率；做錯了的，降低
```

**為什麼要學這個：**
Actor-Critic、PPO 全部建立在這個基礎上。
RLHF（ChatGPT 的訓練方式）用的就是 PPO。

---

### ⬜ 階段 4 — Actor-Critic (A2C)
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

### ⬜ 階段 5 — PPO on CartPole
**核心概念：** Clipped Objective、on-policy vs off-policy

Actor-Critic 的問題：每次更新幅度不好控制，一步走太大就崩掉。
PPO 用 clip 限制每次 policy 的更新幅度。

```
PPO Objective：
  L = min( r_t × A_t,  clip(r_t, 1-ε, 1+ε) × A_t )

  r_t = π_new(a|s) / π_old(a|s)  （新舊 policy 的比值）
  → 比值偏離 1 太多就截斷，保守更新
```

**為什麼是終點：**
PPO 是目前最常用的 RL 演算法，OpenAI、DeepMind 的大多數工作都用它。
RLHF 的 reward model 訓練完之後，用 PPO 微調語言模型。

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
        ├── A2C              ← 階段 4
        └── PPO              ← 階段 5、6
```

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

```bash
pip install numpy gymnasium
```

執行範例：

```bash
python rl_helloworld.py    # 階段 1
python dqn_cartpole.py     # 階段 2
```
