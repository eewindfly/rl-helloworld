"""
RL Hello World 5b — PPO 連續動作版 on Pendulum
================================================
和 ppo_cartpole.py 的關係：PPO 主體（ratio + clip + GAE + K 次複用）完全相同，
唯一的「概念」改動是動作分布 Categorical → Normal（連續動作）。

【這個階段教兩件事】

  (1) 連續動作（本階段的核心概念）
      階段 1~5 的動作全是「離散」的——CartPole 只有左/右，policy 用
      Categorical(softmax) 輸出每個動作的機率。真實控制常是「連續」的
      （施多大力、轉幾度）。解法：policy 改輸出一個連續分布 Normal(mean, std)，
      動作從中取樣。

      ★ 關鍵洞見：PPO 的 ratio = π_new/π_old 只看 log_prob，與動作分布無關。
        所以相對 ppo_cartpole.py，PPO 主體一行不改——這就是要演示的事。

  (2) GAE 是 swing-up 能訓起來的關鍵（驗證階段 4c 那塊拼圖的威力）
      Pendulum 是「把垂下的桿子甩上去」：力矩 ≪ 重力，得先來回擺、累積動量
      幾十步才立得起來——典型的「長程信用分配」。階段 4c 介紹的 GAE（這裡
      沿用，λ=0.95）正是為此而生：把幾十步後才兌現的好結果，用 (γλ)^l 衰減
      後一路傳回現在這一步。

【相比 ppo_cartpole.py 的 diff】

  概念改動（唯一一個）：
    動作分布 Categorical(logits) → Normal(mean, std)
    Actor 輸出 logits(2) → mean(1) + 可學習 log_std；log_prob 對動作維度加總。

  ★ 演算法與 ppo_cartpole.py「完全相同」：ratio、clip、GAE、critic、/N
    正規化、lr、batch 全部一字不差。刻意**不加** advantage 正規化——前面
    每個階段都沒加，加了就等於偷偷改了更新規則，破壞「只換分布」這句話。

  唯一的鬆綁（且只動「訓練預算」、不動演算法）：
    更新次數 250 → 600。Pendulum 比 CartPole 難（swing-up），純粹要更多
    資料才收斂；這不改任何一行更新邏輯，和「多跑幾個 epoch」同性質。
    （實測：no-norm 純 GAE 跑到 600 次更新就能解到 ≈ -250。）

────────────────────────────────────────────────────────────
【λ 開關：一鍵看見「為什麼非 GAE 不可」（本階段最重要的實驗）】

  把 gae_lambda 從 0.95 改成 0.0 重跑——advantage 就退化回階段 4b 的
  「單步 TD δ」。結果（實測，兩者都跑滿 600 次更新、其餘設定完全相同）：

    gae_lambda = 0.95（GAE）  → return 從 ≈ -1500 一路爬到 ≈ -256（學會！）
    gae_lambda = 0.0 （單步TD）→ return 卡在 ≈ -1365（隨機水準，學不起來）

  ⚠️ 關鍵對照：「其他全給齊（同樣 600 次更新、同樣沒有正規化），只把 GAE
     關掉」，它就學不起來。
     ⇒ 證明讓 Pendulum 能學的決定性因素就是 GAE（λ>0 的多步信用分配）。
       這就是階段 4c 那根 λ 旋鈕的真正威力。

  （這也呼應歷史：真實 PPO 從 2017 原版就標配 GAE。階段 5 + 5b 合起來，
    才是「clip + GAE」這個完整、真實的 PPO。）

────────────────────────────────────────────────────────────
【Pendulum-v1 這個環境】

  觀測 (3 維)：[cos θ, sin θ, 角速度]
  動作 (1 維)：連續力矩 ∈ [-2, 2]
  reward    ：-(θ² + 0.1·θ̇² + 0.001·力矩²)，越接近「正上方靜止」越接近 0
  ⚠️ 沒有提早終止：每條 episode 固定 200 步才 truncated，terminated 恆 False
     → TD 殘差 δ 的 V(s') 永遠 bootstrap（GAE 的反向掃描在 done 邊界重置即可）。
  沒有官方解門檻；隨機 ≈ -1200~-1600，學會甩上去並穩住 ≈ -150~-250。

【連續動作的 log_prob】
  離散：log π = log(softmax(logits)[a])
  連續：log π = Normal(mean, std).log_prob(a)（對動作各維相加 → .sum(-1)）
  std 來自「與狀態無關的可學習 log_std」（CleanRL/SpinningUp 標準最小做法）：
  初期 std≈1 多探索，學會後自動縮小。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import gymnasium as gym
from actor_critic import CriticNetwork   # Critic 架構完全相同，改輸入維度 3 即複用


# ─────────────────────────────────────────────────────────
#  連續動作的 Actor：輸出高斯分布的 mean + 可學習 log_std
#  （這是相對 ppo_cartpole.py 唯一真正改動的地方）
# ─────────────────────────────────────────────────────────

class GaussianActorNetwork(nn.Module):
    """
    連續動作 Actor：輸入狀態，輸出高斯分布的「平均 mean」。
    架構：輸入(3) → 隱藏層(64, ReLU) → 輸出(1, mean，線性)

    和離散版 PolicyNetwork 的差異：
      離散：輸出 2 個 logits → Categorical → 機率（加總為 1）
      連續：輸出 1 個 mean  → 配 log_std → Normal(mean, std)
    """
    def __init__(self, input_dim=3, hidden_dim=64, action_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        # log_std 初始為 0 → std = exp(0) = 1（適中的初始探索幅度）
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        """狀態 → (mean, std)。std 由可學習的 log_std 取 exp 得到。"""
        mean = self.net(x)                  # (T, action_dim)
        std  = torch.exp(self.log_std)      # (action_dim,)，broadcast 到 (T, action_dim)
        return mean, std

    @torch.no_grad()
    def predict_mean(self, state):
        """單筆推論：numpy 狀態 → numpy 平均動作（給展示用的確定性 policy）"""
        x = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        mean, _ = self.forward(x)
        return mean[0].numpy()


# ─────────────────────────────────────────────────────────
#  PPO Agent（與 ppo_cartpole.py 演算法完全相同，僅分布相關處改動）
# ─────────────────────────────────────────────────────────

class PPOAgent:
    def __init__(self, action_low, action_high):
        self.actor  = GaussianActorNetwork()
        self.critic = CriticNetwork(input_dim=3)   # Pendulum 觀測是 3 維

        self.gamma      = 0.99
        self.gae_lambda = 0.95   # ★ 改成 0.0 → 退回單步 TD → 學不起來（見檔頭 λ 開關）
        # lr 對齊 ppo_cartpole.py（0.001/0.005），不另調——強調「只改了分布」。
        self.actor_lr  = 0.001
        self.critic_lr = 0.005

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=self.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # ── PPO 核心超參（與 ppo_cartpole.py 相同）──
        self.clip_eps = 0.2
        self.k_epochs = 4

        # 動作邊界（送進環境前用來 clip）
        self.action_low  = action_low
        self.action_high = action_high

        self._states      = []
        self._actions     = []
        self._rewards     = []
        self._next_states = []
        self._terminateds = []
        self._dones       = []

    @torch.no_grad()
    def choose_action(self, state):
        """
        從高斯分布取樣連續動作。回傳 (raw_action, env_action)：
          raw_action：分布原始取樣值，存進 buffer 給 log_prob 用（不裁剪）
          env_action：clip 到 [low, high] 後送進環境
        分開的理由：拿「裁剪後」的動作回算 log_prob 會讓邊界處密度失真，
        ratio 的 important sampling 就不準。標準做法：log_prob 用原始值，
        裁剪只發生在丟給環境那一步。
        """
        x = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        mean, std  = self.actor(x)
        raw_action = Normal(mean, std).sample()[0].numpy()        # (action_dim,) 未裁剪
        env_action = np.clip(raw_action, self.action_low, self.action_high)
        return raw_action, env_action

    def store(self, state, raw_action, reward, next_state, terminated, done):
        self._states.append(state)
        self._actions.append(raw_action)   # ⚠️ 存未裁剪的 raw_action
        self._rewards.append(reward)
        self._next_states.append(next_state)
        self._terminateds.append(terminated)
        self._dones.append(done)

    def compute_gae(self, deltas, dones):
        """反向掃描把單步 δ 累積成 GAE（和 ppo_cartpole.py / 4c 完全相同）。"""
        T = len(deltas)
        advantages = torch.zeros(T)
        last = 0.0
        for t in reversed(range(T)):
            if dones[t]:
                last = 0.0
            last = deltas[t] + self.gamma * self.gae_lambda * last
            advantages[t] = last
        return advantages

    def update(self):
        states      = torch.as_tensor(np.array(self._states),      dtype=torch.float32)  # (T,3)
        actions     = torch.as_tensor(np.array(self._actions),     dtype=torch.float32)  # (T,1) 連續！
        rewards     = torch.as_tensor(np.array(self._rewards),     dtype=torch.float32)  # (T,)
        next_states = torch.as_tensor(np.array(self._next_states), dtype=torch.float32)  # (T,3)
        terminateds = torch.as_tensor(np.array(self._terminateds), dtype=torch.bool)     # (T,) 恆 False
        N = int(np.sum(self._dones))

        # ══════════════════════════════════════════════════════════
        #  Phase 1：更新前，用「舊策略 / 舊 critic」算好並凍結
        #           （和 ppo_cartpole.py 逐字相同，僅分布 Categorical → Normal）
        # ══════════════════════════════════════════════════════════
        with torch.no_grad():
            # (a) old_log_prob = log π_old(a|s)（★ 唯一分布改動：Normal + 對動作維度加總）
            old_mean, old_std = self.actor(states)                            # (T,1),(1,)
            old_log_probs = Normal(old_mean, old_std).log_prob(actions).sum(axis=-1)  # (T,)

            # (b) 每步 TD 殘差 δ（Pendulum 恆不 terminated → 永遠 bootstrap）
            values      = self.critic(states)                    # (T,)  V(s_t)
            next_values = self.critic(next_states)               # (T,)  V(s_{t+1})
            next_values[terminateds] = 0.0
            deltas      = rewards + self.gamma * next_values - values  # (T,)  δ_t

            # (c) advantage = GAE（沿用 4c/5）；critic target = λ-return
            #     ⚠️ 沒有 advantage 正規化——演算法與 ppo_cartpole.py 完全相同，
            #        只換了動作分布。Pendulum 較難純粹靠「多跑幾次」解決（見 train）。
            advantages = self.compute_gae(deltas, self._dones)   # (T,)  Â_t^GAE
            returns    = advantages + values                     # (T,)  λ-return（critic target）

        # ══════════════════════════════════════════════════════════
        #  Phase 2：同一批資料，重複做 K 次完整 update
        #           （和 ppo_cartpole.py 唯一差別仍是 new_logp 用 Normal）
        # ══════════════════════════════════════════════════════════
        for _ in range(self.k_epochs):
            mean, std = self.actor(states)                            # (T,1),(1,)
            new_logp  = Normal(mean, std).log_prob(actions).sum(axis=-1)  # (T,)

            ratio = torch.exp(new_logp - old_log_probs)              # r_t(θ) = π_new/π_old
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps,
                                       1 + self.clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).sum() / N          # −L^CLIP(θ)
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            v = self.critic(states)                                  # (T,)  V_φ(s_t)
            critic_loss = F.mse_loss(v, returns)                     # (V_φ(s_t) − λ-return)²
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

        self._states      = []
        self._actions     = []
        self._rewards     = []
        self._next_states = []
        self._terminateds = []
        self._dones       = []


# ─────────────────────────────────────────────────────────
#  訓練迴圈
# ─────────────────────────────────────────────────────────

def train():
    env = gym.make("Pendulum-v1")
    action_low  = float(env.action_space.low[0])
    action_high = float(env.action_space.high[0])
    agent = PPOAgent(action_low, action_high)

    BATCH_EPISODES = 4     # 每收集 4 條軌跡才更新一次（與離散版相同）
    NUM_UPDATES    = 600   # Pendulum 較難，比 CartPole(250) 多跑——唯一的鬆綁，
                           # 純訓練預算、不改演算法（演算法與 ppo_cartpole.py 完全相同）
    EPISODES       = BATCH_EPISODES * NUM_UPDATES
    scores         = []

    print("=" * 60)
    print("  RL Hello World 5b — PPO 連續動作版 on Pendulum [PyTorch]")
    print("=" * 60)
    print(f"\nclip ε = {agent.clip_eps}   K_epochs = {agent.k_epochs}   "
          f"GAE λ = {agent.gae_lambda}")
    print("核心改動：動作分布 Categorical → Normal(mean, std)（連續動作）")
    print("PPO 主體：ratio + clip + GAE + 同批 K 次複用（與 ppo_cartpole.py 相同）")
    print("目標    ：return 從 ≈ -1500 爬向 ≈ -250（學會把桿子甩上去並穩住）")
    print("λ 實驗  ：把 gae_lambda 改 0.0 → 退回單步 TD → 學不起來（見檔頭）")
    print(f"\n開始訓練 {EPISODES} 個 episodes...\n")

    ep_in_batch = 0
    for episode in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0

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
            avg_score = np.mean(scores[-50:])
            solved = "✓ 學會了！" if avg_score >= -250 else ""
            print(f"Episode {episode+1:4d} | 近50回合平均 return: {avg_score:8.1f}  {solved}")

    env.close()

    # ── 展示（用確定性的 mean 動作）──
    print("\n" + "=" * 60)
    print("  訓練完成！跑 5 次展示（用 mean 動作）")
    print("=" * 60)
    env = gym.make("Pendulum-v1")
    for trial in range(5):
        state, _ = env.reset()
        total_reward = 0
        while True:
            env_action = np.clip(agent.actor.predict_mean(state), action_low, action_high)
            state, reward, terminated, truncated, _ = env.step(env_action)
            total_reward += reward
            if terminated or truncated:
                break
        result = "✓ 穩住了！" if total_reward >= -250 else ""
        print(f"  Trial {trial+1}：return = {total_reward:8.1f}  {result}")
    env.close()

    # ── 對比總結：離散 PPO vs 連續 PPO ──
    print("\n" + "=" * 60)
    print("  離散 PPO (CartPole) vs 連續 PPO (Pendulum)")
    print("=" * 60)
    rows = [
        ("動作空間",     "離散（左/右）",            "連續（力矩 ∈ [-2,2]）"),
        ("Actor 輸出",   "logits(2)",                "mean(1) + log_std"),
        ("動作分布",     "Categorical(logits)",      "Normal(mean, std)"),
        ("PPO 主體",     "ratio+clip+GAE+K複用",     "完全相同（一行不改）"),
        ("adv 正規化",   "無",                        "無（保持最小 diff）"),
        ("更新次數",     "250",                       "600（純訓練預算，較難）"),
    ]
    print(f"  {'':12s} {'離散 PPO':28s} {'連續 PPO':28s}")
    print("  " + "-" * 70)
    for label, a, p in rows:
        print(f"  {label:12s} {a:28s} {p:28s}")
    print("\n核心洞見：")
    print("  1. PPO 的 ratio + clip 與動作分布無關——換成高斯，PPO 主體一行不改。")
    print("  2. swing-up 能學起來靠的是 GAE（λ=0.95）；把 λ 改 0（單步 TD），")
    print("     其餘設定全給齊（一樣沒正規化、一樣跑 600 次），依然學不起來——GAE 才是關鍵。")

    return agent


def demo(agent, episodes=None):
    """開視窗展示連續 policy（用 mean 動作）。utils.demo 寫死 CartPole，故這裡自帶。"""
    print("\n開啟視覺化視窗，按 Ctrl+C 結束...")
    env = gym.make("Pendulum-v1", render_mode="human")
    low  = float(env.action_space.low[0])
    high = float(env.action_space.high[0])
    ep = 0
    try:
        while episodes is None or ep < episodes:
            state, _ = env.reset()
            total_reward = 0
            while True:
                action = np.clip(agent.actor.predict_mean(state), low, high)
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    print(f"  Episode {ep+1}：return = {total_reward:8.1f}")
                    break
            ep += 1
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    agent = train()
    demo(agent)
