# RL Hello World — 強化學習學習計劃

從零開始，用最小可執行的範例理解每個核心概念。
每個階段都有對應的程式碼，跑起來、改參數、看結果。

---

## 進度總覽

| # | 檔案 | 演算法 | 環境 | 狀態 |
|---|------|--------|------|------|
| 1 | `q_learning_gridworld.py` | Q-Learning | GridWorld (4×4) | ✅ 完成 |
| 2 | `dqn_cartpole.py` | DQN | CartPole-v1 | ✅ 完成 |
| 3 | `pg_cartpole.py` | Policy Gradient (REINFORCE) | CartPole-v1 | ✅ 完成 |
| 4 | `actor_critic.py` | Actor-Critic (A2C, MC) | CartPole-v1 | ✅ 完成 |
| 4b | `actor_critic_td.py` | Actor-Critic (A2C, TD) | CartPole-v1 | ✅ 完成 |
| 4c | `actor_critic_gae.py` | Actor-Critic (A2C, GAE) | CartPole-v1 | ✅ 完成 |
| 5 | `ppo_cartpole.py` | PPO (PPO-Clip + GAE) | CartPole-v1 | ✅ 完成 |
| 5b | `ppo_pendulum.py` | PPO 連續動作 (Normal + GAE) | Pendulum-v1 | ✅ 完成 |
| 6a | `ppo_custom_env_schedule.py` | PPO 離散（沿用 5） | 自訂排程環境 | ✅ 完成 |
| 6b | `ppo_custom_env_trading.py` | PPO 連續（沿用 5b） | 自訂交易環境 | ✅ 完成 |

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

這版是 advantage 光譜的一個端點（λ=0）——下一階段 4c 的 GAE 會把它和
階段 4 的 MC（λ=1）連起來。

---

### ✅ 階段 4c — Actor-Critic (A2C，GAE 版)
**核心概念：** GAE、λ 在 bias↔variance 間的折衷、把 MC 與 TD 連成光譜

GAE（Generalized Advantage Estimation, Schulman 2015）不是憑空的新東西，
它就是**把你已經做過的階段 4（MC）和階段 4b（TD）統一起來的那根旋鈕**：

```
Â_t = Σ_{l≥0} (γλ)^l · δ_{t+l}     δ_t = r + γV(s') - V(s)

  λ = 0  → Â_t = δ_t            （退回階段 4b 的單步 TD，高 bias、方差低）
  λ = 1  → Â_t = G_t - V(s)     （退回階段 4 的 MC，無 bias、方差高）
  λ≈0.95 → 兩者折衷（多步加權，本檔用這個）
```

直覺：單步 δ 只看下一步；GAE 把「後面好幾步的 δ」用 (γλ)^l 衰減後加總，
讓「現在這個動作」沾得到「幾步後才兌現的好結果」——這就是長程信用分配。

```
相比 actor_critic_td.py 的唯一改動：
  advantage = δ（單步）   →   advantage = Σ(γλ)^l·δ（反向掃描累積）
  critic target 同步從「單步 td_target」換成「λ-return = Â + V」
  （維持本系列不變式：critic target = advantage + V(s)）
```

> ⚠️ CartPole 上 GAE 和 TD 表現差不多（dense +1、單步 δ 本來就夠用）。
> GAE 的威力要到「長程信用分配」的任務才看得出來——下一個丟上 Pendulum
> 的連續控制階段，你會親眼看到 λ=0 學不起來、λ=0.95 才行。本階段先把
> GAE 這個工具乾淨地介紹清楚。

> **歷史定位：** GAE(2015) 早於 PPO(2017)，而 PPO 從原版起就把 GAE 當標配。
> 先學 GAE 再進 PPO，順序與真實研究史一致——PPO 是踩在 GAE 上長出來的。

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
advantage **沿用階段 4c 的 GAE（λ=0.95）**，因為 GAE 是 advantage 估計法、
和 clip 完全正交，已在 4c 單獨介紹過，這裡直接用即可——這也才是真實
PPO 的標準配置。刻意**不放** entropy bonus、共享網路、advantage 正規化，
**也不切 minibatch**（minibatch 是正交的 SGD 工程技巧，AC 一樣能用，非
PPO 核心），讓 diff 精準隔離出「PPO = A2C(GAE) + clip」。為求乾淨對照，
batch（4）、更新次數（250）、actor 正規化（`.sum()/N`）、lr（actor 0.001 /
critic 0.005）、GAE λ（0.95）全部對齊階段 4c。

```
相比 actor_critic_gae.py 的唯一改動（只有這三點）：
  1. 更新前凍結 old_log_prob（π_old）與 advantage(GAE) / λ-return
  2. 目標  log π · A  →  min(ratio·A, clip(ratio)·A)，ratio 做 important sampling
  3. 外層加「整批複用 K 次」迴圈，重複用同一批資料（不切 minibatch）
  → advantage 的算法（GAE）一字不改，純粹多了 clip + 同批複用。
```

> 細節：用 `/N`（軌跡數）正規化而非 `.mean()`，與整個系列一致——這讓
> PPO「第一個 epoch」的梯度精確等於 AC(GAE)（ratio=1 時 ∇(ratio·A)=∇log π·A），
> PPO 因此是 AC(GAE) 不打折的乾淨超集。

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

### ✅ 階段 5b — PPO 連續動作版 on Pendulum
**核心概念：** 連續動作空間、高斯 policy（Normal）；並驗證 GAE 對 swing-up 的威力

**核心概念是「連續動作」。** 階段 1~5 的動作全是「離散」的（CartPole 只有
左/右，policy 用 Categorical softmax）。真實控制常是「連續」的（施多大力、
轉幾度）。解法：policy 改輸出一個連續分布 `Normal(mean, std)`，動作從中取樣。

```
概念改動（相比 ppo_cartpole.py，唯一一個）：
  Actor 輸出 logits(2)   →  mean(1) + 可學習 log_std
  Categorical(logits)    →  Normal(mean, std)；log_prob 對動作維度加總
  ratio / clip / GAE / critic / lr / batch / 正規化 →  PPO 主體一字不差
```

> ★ 關鍵洞見：PPO 的 ratio = π_new/π_old 只看 log_prob，**與動作分布無關**。
> 換成高斯，PPO 主體完全不動——連續控制（機器人、MuJoCo）用的就是這套。

**順帶驗證 GAE 的威力（swing-up）。** Pendulum 是「把垂下的桿子甩上去」：
力矩 ≪ 重力，得先來回擺、累積動量幾十步才立得起來——典型的「長程信用分配」。
階段 4c 的 GAE（這裡沿用，λ=0.95）正是為此而生。實測 return 從 ≈ -1500
一路爬到 ≈ -256（學會把桿子甩上並穩住）。

```
⚠️ 唯一的鬆綁（且只動「訓練預算」、不動演算法）：更新次數 250 → 600。
   Pendulum 比 CartPole 難，純粹要更多資料才收斂——和「多跑幾個 epoch」同性質。
   刻意「不加」advantage 正規化：前面每個階段都沒加，加了等於偷改更新規則，
   破壞「只換了動作分布」這句話。實測純 GAE 跑滿 600 次就能解到 ≈ -250。
```

> **λ 開關（本階段最重要的實驗）：** 把 `gae_lambda` 從 0.95 改成 **0.0**，
> advantage 就退回階段 4b 的單步 TD。實測（兩者都跑滿 600 次、其餘設定完全相同）：
> λ=0.95 → 爬到 ≈ -256（學會）；λ=0 → 卡在 ≈ -1365（隨機水準）。
> ⇒ 在「其他全給齊、只把 GAE 關掉」下它就學不起來，證明決定性因素就是 **GAE**。
> 這就是階段 4c 那根 λ 旋鈕的真正威力，也是「clip + GAE」這個完整真實 PPO 的價值。

---

### ✅ 階段 6 — PPO on 自訂環境
**核心概念：** Gymnasium 介面、reward shaping、自己設計問題

前面 1~5b 的主角都是「演算法」，環境（CartPole / Pendulum）是別人寫好的。
真實工作裡剛好相反：**演算法直接拿現成的 PPO，難的是把你的問題包成一個
Gymnasium 環境**——定義 state / action / reward，尤其是 reward shaping。
這一階段親手體驗：「定義問題」比「改演算法」更需要動腦。做了兩個自訂環境，
一個離散、一個連續，各複用前面一條 PPO。

#### ✅ 階段 6a — 多機台排程（離散，沿用階段 5 的 PPO）

一條條進來的 job 各有大小，即時決定丟到 M 台機器哪一台，目標讓 makespan
（最忙那台的總負載）最小——也就是負載平衡。

```
state ：[當前 job 大小, 各機台負載(去均值), 進度比例]   → M+2 維
action：丟到哪一台（離散 M 選 1）
reward：-(這步造成的 makespan 增量)
```

**reward shaping 的關鍵（本檔最該看的地方）：**
每步給「makespan 增量」的負值，整條 episode 加起來會望遠鏡式抵銷成
`−最終 makespan`——和「只在最後給 −makespan」等價，但**每步都有訊號**
（dense reward），PPO 學得快、critic 好估。沒人教它「挑最閒的丟」，它從
這個 reward 自己學出負載平衡，訓練後 makespan 逼近貪婪(最閒優先)啟發式。

```
與階段 5 的 diff：PPO 演算法（ratio+clip+GAE+K 次複用）一行不改，
  只 (1) 換環境，(2) 因 obs=5/act=3 而調 Actor/Critic 的「輸入輸出維度」
  （只動網路 in/out 尺寸，演算法邏輯沒碰）。
```

#### ✅ 階段 6b — 均值回歸交易（連續，沿用階段 5b 的 PPO）

對一個均值回歸（OU 過程）的價格做多空：價格被拉回均值，偏高該放空、偏低該
做多。動作是「持有多少部位」——連續的，所以用 5b 的高斯 policy。

```
state ：[價格(去均值), 上一步價格變動, 目前部位]  → 3 維（與 Pendulum 相同）
action：目標部位 q ∈ [-1,1]                   → 1 維連續
reward：pnl − 交易成本 = q·(下一步價格變動) − c·|換手|
```

**★ 連網路維度都不用換：** obs 剛好 3 維、action 1 維，與 Pendulum 介面完全
相同，於是**直接 import 階段 5b 的 `PPOAgent`，整個 agent 一行不改**，只傳進
動作邊界 [-1,1]。這份檔案新增的「只有環境」——把階段 6 的精神釘到最死。
沒人教它「價高放空」，它從「pnl − 成本」自己學出均值回歸交易，學到的
「價格→部位」反應函數呈負斜率，return 從 ≈0 逼近先知規則。

> **兩種複用對照：** 6a 還要換一下網路維度；6b 連這都省了。一起說明：
> 真實世界用 PPO，動作離散就接離散版、連續就接連續版，**演算法是現成的，
> 你出力的地方永遠是「定義問題」**。

> ⚠️ 6b 是教學用的合成價格（OU 均值回歸），不是真實市場；真實價格遠更接近
> 隨機漫步、訊號弱得多。這裡的重點是「如何把問題包成環境 + reward shaping」，
> 不是可獲利的交易策略。

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
        ├── A2C (MC)         ← 階段 4    advantage：G_t（λ=1 端點）
        ├── A2C (TD)         ← 階段 4b   advantage：單步 δ（λ=0 端點）
        ├── A2C (GAE)        ← 階段 4c   advantage：GAE（λ 旋鈕，統一 MC↔TD）
        └── PPO = A2C(GAE) + clip
            ├── 離散 (CartPole)      ← 階段 5
            ├── 連續動作 (Normal)    ← 階段 5b（Pendulum，GAE 讓 swing-up 學得起來）
            └── 自訂環境             ← 階段 6（演算法現成，重點在定義問題）
                ├── 離散：排程       ← 階段 6a（沿用 5，只換網路維度）
                └── 連續：交易       ← 階段 6b（沿用 5b，agent 一行不改）
```

> **關於「嚴格最小 diff」**：policy-gradient 鏈（pg → AC(MC) → AC(TD) →
> AC(GAE) → PPO）刻意把 batch（4）、更新次數（250）、總 episodes（1000）、
> actor 正規化（`.sum()/N`）、gamma（0.99）、網路架構全部對齊，好讓階段間的
> diff 只剩「核心概念」。**唯一沒被隔離的變數是 learning rate**：每階段各自
> 重調（pg 0.01 → AC(MC) 0.0005/0.001 → AC(TD)/AC(GAE)/PPO 0.001/0.005）。
> 原因是 lr 與演算法本質耦合（return / advantage / TD δ 的天然尺度不同，TD 又
> 特別吃 critic 準度），屬「必要的重調」而非正交超參，故保留但明示。
>
> **階段 5b（Pendulum）的唯一鬆綁**：它連 lr / batch / 正規化都對齊離散 PPO
> （0.001/0.005），演算法一字不差，只換了動作分布。Pendulum 較難，唯一放寬的
> 是更新次數 250→600——但這只動「訓練預算」、不動更新規則（和「多跑幾個
> epoch」同性質）。刻意**不加** advantage 正規化：那會偷改更新規則，破壞最小
> diff；實測純 GAE 跑滿 600 次就能解，根本不需要它。

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
| GAE | 用 λ 把單步 TD（λ=0）與 MC（λ=1）的 advantage 連成光譜，把長程訊號傳回來（階段 4c） |
| λ-return | GAE 對應的 critic target：Â + V(s)，即 TD(λ) 的目標（階段 4c） |
| 連續動作 | 動作是連續實數（如力矩），policy 改輸出 `Normal(mean, std)` 取樣（階段 5b） |
| Gymnasium 介面 | 把問題包成 `reset` / `step` / `observation_space` / `action_space` 的標準環境（階段 6） |
| Reward shaping | 在不改最終目標下，把稀疏終局獎勵拆成密集逐步獎勵，讓 agent 學得快（階段 6a） |

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
python q_learning_gridworld.py     # 階段 1  Q-Learning
python dqn_cartpole.py      # 階段 2  DQN
python pg_cartpole.py       # 階段 3  Policy Gradient
python actor_critic.py      # 階段 4  A2C (MC)
python actor_critic_td.py   # 階段 4b A2C (TD)
python actor_critic_gae.py  # 階段 4c A2C (GAE)
python ppo_cartpole.py      # 階段 5  PPO (+ GAE)
python ppo_pendulum.py      # 階段 5b PPO 連續動作（Pendulum，GAE 讓 swing-up 學得起來）
python ppo_custom_env_schedule.py  # 階段 6a 自訂排程環境（離散，沿用階段 5）
python ppo_custom_env_trading.py   # 階段 6b 自訂交易環境（連續，沿用階段 5b）
```
