"""
RL Hello World 6b — PPO on 自訂環境：均值回歸交易（連續動作）
=============================================================
這是「階段 6」的第二個自訂環境。和 6a（排程）一樣，重點是**定義問題**；
不同的是這裡的動作是「連續」的（要持有多少部位），所以沿用階段 5b 的
連續 PPO（高斯 policy）。

【★ 與前面階段的關係：PPO 演算法「整個檔案一行不改」】

  本檔的觀測剛好是 3 維、動作剛好是 1 維——和 Pendulum 完全相同的介面。
  於是我們可以 **直接 import 階段 5b 的 `PPOAgent`（ppo_pendulum.py），
  連網路維度都不用換**，唯一傳進去的只有動作邊界 [-1, 1]。

    → 階段 6a 還要換一下網路維度；6b 連這個都省了：
      PPO 的程式碼 100% 原封不動，這份檔案新增的「只有環境」。
      這就是階段 6 想釘死的一句話：**演算法是現成的，你的工作是定義問題。**

【本檔的問題：對一個「均值回歸」的價格做多空】

  價格 p 跟著 Ornstein–Uhlenbeck（均值回歸）過程走：
      p_{t+1} = p_t + κ·(μ − p_t) + σ·ε          （μ=0：價格繞著 0 上下擺）
  直覺：價格被拉回均值。p 偏高 → 預期會跌 → 該『放空』；p 偏低 → 預期會漲
  → 該『做多』。所以存在一個可學的訊號（這正是要 agent 發現的東西）。

    state  ：[價格 p（已去均值）, 上一步價格變動, 目前部位] → 3 維（同 Pendulum）
    action ：目標部位 q ∈ [-1, 1]（−1 全空、0 空手、+1 全多）→ 1 維連續
    reward ：pnl − 交易成本
             pnl  = 部位 × 下一步價格變動 = q·(p_{t+1} − p_t)
             成本 = c·|新部位 − 舊部位|     （換手要付手續費，抑制亂頻繁交易）

  最佳直覺策略：部位 q ≈ −sign(p)（價高放空、價低做多），
  因為 E[p_{t+1} − p_t] = κ(μ − p) = −κ·p。本檔會把 PPO 跟這個
  「先知規則」與「buy & hold」對照，看它學得多接近。

【reward shaping 的眉角（這份環境真正要動腦的地方）】

  - 用「下一步」的價格變動算 pnl：你在 t 設好部位，賺的是 t→t+1 這段的波動。
  - 加交易成本：不然 agent 會每步在 ±1 之間亂跳。成本逼它『有把握才換手』。
  - 沒有 advantage 正規化、沒有 reward 縮放：和整個系列一致，保持最小 diff。
    （訊號夠強，純 GAE 就學得起來；要動的只有訓練預算。）
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── 直接沿用階段 5b 的連續 PPO：整個 agent 原封不動（obs=3、action=1 完全吻合）──
from ppo_pendulum import PPOAgent


# ═════════════════════════════════════════════════════════════
#  自訂環境：把「均值回歸交易」包成標準 Gymnasium 介面
#  —— 這份 class 就是階段 6b 唯一的新東西
# ═════════════════════════════════════════════════════════════

class MeanRevertTradingEnv(gym.Env):
    """
    單一資產、均值回歸價格的多空交易環境。

    一個 episode = horizon 步。每步 agent 設定目標部位 q∈[-1,1]，
    賺取 q×(下一步價格變動) 減去換手成本。

    Gymnasium 介面：
      observation_space：3 維 Box（價格、上一步價格變動、目前部位）
      action_space     ：1 維 Box ∈ [-1, 1]（目標部位）
      reset / step     ：標準 RL 互動協定
    """
    metadata = {"render_modes": []}

    def __init__(self, horizon=100, kappa=0.15, sigma=0.3, cost=0.005, seed=None):
        super().__init__()
        self.horizon = horizon
        self.kappa   = kappa          # 均值回歸速度（越大拉回越快、訊號越強）
        self.sigma   = sigma          # 價格噪音
        self.cost    = cost           # 每單位換手的交易成本
        # 平穩分布標準差（OU 理論值），拿來正規化觀測裡的價格
        self._price_std = sigma / np.sqrt(max(1e-8, 2 * kappa - kappa**2))

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # 從平穩分布抽起始價，部位歸零
        self.price     = float(self._rng.normal(0.0, self._price_std))
        self.position  = 0.0
        self.last_price_move = 0.0
        self.t         = 0
        return self._obs(), {}

    def _obs(self):
        # 價格用平穩標準差正規化 → 大致落在 [-3,3]，數值穩定
        # last_price_move = Δp（市場價格變動），不是 agent 的 reward
        return np.array([self.price / self._price_std,
                         self.last_price_move,
                         self.position], dtype=np.float32)

    def step(self, action):
        new_position = float(np.clip(action, -1.0, 1.0).item())

        # 價格前進一步（OU 均值回歸，μ=0）
        eps        = self._rng.normal(0.0, 1.0)
        next_price = self.price + self.kappa * (0.0 - self.price) + self.sigma * eps
        price_move = next_price - self.price

        # reward = 部位賺到的 pnl − 換手成本
        pnl    = new_position * price_move
        cost   = self.cost * abs(new_position - self.position)
        reward = pnl - cost

        # 推進狀態
        self.last_price_move = price_move
        self.price    = next_price
        self.position = new_position
        self.t       += 1

        truncated  = (self.t >= self.horizon)   # 固定長度，無提早終止（同 Pendulum）
        terminated = False
        return self._obs(), reward, terminated, truncated, {}


# ═════════════════════════════════════════════════════════════
#  對照策略（給「學得好不好」一把尺）
# ═════════════════════════════════════════════════════════════

def _avg_episode_return(env, action_fn, episodes=200, seed=0):
    """跑若干局，回傳平均「每局累積 reward」。action_fn: (env, obs) → 部位陣列。"""
    s = seed
    total = 0.0
    for _ in range(episodes):
        obs, _ = env.reset(seed=s); s += 1
        ep = 0.0
        while True:
            a = action_fn(env, obs)
            obs, r, term, trunc, _ = env.step(a)
            ep += r
            if term or trunc:
                break
        total += ep
    return total / episodes


def oracle_action(env, obs):
    """先知規則（參考上界）：部位 = clip(−price/price_std)，價高放空、價低做多。"""
    return np.array([np.clip(-obs[0], -1.0, 1.0)], dtype=np.float32)


def buy_and_hold_action(env, obs):
    """一路做多（參考點）：μ=0 的均值回歸下，長期期望≈0。"""
    return np.array([1.0], dtype=np.float32)


# ═════════════════════════════════════════════════════════════
#  訓練迴圈（和 ppo_pendulum.py 的 train 幾乎一樣，只是換了 env）
# ═════════════════════════════════════════════════════════════

def train():
    env = MeanRevertTradingEnv(horizon=100, seed=0)
    # ★ 連續 PPO agent 原封不動繼承階段 5b；只把動作邊界改成 [-1, 1]
    agent = PPOAgent(action_low=-1.0, action_high=1.0)

    BATCH_EPISODES = 4
    NUM_UPDATES    = 500     # 比 CartPole(250) 多；交易訊號帶噪音，純訓練預算
    EPISODES       = BATCH_EPISODES * NUM_UPDATES
    scores         = []

    print("=" * 64)
    print("  RL Hello World 6b — PPO 自訂環境：均值回歸交易（連續）[PyTorch]")
    print("=" * 64)
    print("\n演算法 ：完全繼承階段 5b 的連續 PPO（高斯 policy），整個 agent 一行不改")
    print("新東西 ：MeanRevertTradingEnv（定義問題）+ pnl−成本 的 reward shaping")
    print(f"clip ε = {agent.clip_eps}   K_epochs = {agent.k_epochs}   GAE λ = {agent.gae_lambda}")

    # 訓練前的參考點
    ref = MeanRevertTradingEnv(horizon=100)
    r_bh     = _avg_episode_return(ref, buy_and_hold_action, episodes=300, seed=10_000)
    r_oracle = _avg_episode_return(ref, oracle_action,        episodes=300, seed=10_000)
    print(f"\n參考點（每局平均累積 reward，越大越好）：")
    print(f"  buy & hold（一路做多） ≈ {r_bh:+.3f}")
    print(f"  先知規則（價高空/價低多）≈ {r_oracle:+.3f}  ← 參考上界")
    print(f"目標   ：PPO 從 ≈0 學到逼近先知 ≈ {r_oracle:+.3f}")
    print(f"\n開始訓練 {EPISODES} 個 episodes...\n")

    ep_in_batch = 0
    for episode in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0.0
        while True:
            raw_action, env_action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(env_action)
            done = terminated or truncated
            agent.store(state, raw_action, reward, next_state, terminated, done)
            total_reward += reward
            state = next_state
            if done:
                break
        scores.append(total_reward)
        ep_in_batch += 1
        if ep_in_batch == BATCH_EPISODES:
            agent.update()
            ep_in_batch = 0

        if (episode + 1) % 100 == 0:
            avg = np.mean(scores[-50:])
            tag = "✓ 逼近先知！" if avg >= r_oracle * 0.8 else ""
            print(f"Episode {episode+1:4d} | 近50局平均 return: {avg:+.3f}  {tag}")

    env.close()

    # ── 評估：用確定性 mean 部位 ──
    print("\n" + "=" * 64)
    print("  訓練完成！評估學到的 policy（300 局平均 return）")
    print("=" * 64)
    eval_env = MeanRevertTradingEnv(horizon=100)
    def learned_action(env, obs):
        return np.clip(agent.actor.predict_mean(obs), -1.0, 1.0)
    r_learned = _avg_episode_return(eval_env, learned_action, episodes=300, seed=10_000)

    print(f"\n  buy & hold      ：return ≈ {r_bh:+.3f}")
    print(f"  PPO 學到的策略  ：return ≈ {r_learned:+.3f}")
    print(f"  先知規則        ：return ≈ {r_oracle:+.3f}  ← 參考上界")

    # 抽查學到的「價格 → 部位」對應，看是否呈現負相關（價高放空）
    print("\n  學到的反應函數（價格 p → 目標部位 q，期望為負斜率）：")
    for p_norm in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        obs = np.array([p_norm, 0.0, 0.0], dtype=np.float32)
        q = float(np.clip(agent.actor.predict_mean(obs), -1.0, 1.0).item())
        print(f"    p = {p_norm:+.1f} σ  →  部位 q = {q:+.2f}")

    print("\n核心洞見：")
    print("  1. PPO 程式碼 100% 沿用階段 5b——本階段唯一新增的就是『環境』。")
    print("  2. 沒人教它『價高放空』；它從『pnl − 成本』這個 reward 自己學出均值回歸交易。")
    print("  3. 動作是連續的（部位大小），所以用 5b 的高斯 policy——這也說明")
    print("     連續 PPO 不只用在機器人控制，定價/部位這類問題一樣套得上。")
    return agent


if __name__ == "__main__":
    train()
