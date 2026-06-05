"""
RL Hello World 5b — PPO 連續動作版 on Pendulum（兼「極簡 PPO 的天花板」反例）
================================================
和 ppo_cartpole.py 的關係：演算法「完全相同」，只換了動作分布。

★ 這個階段有兩個身份，請務必都讀懂：
  (1) 正面教材：示範「連續動作」——把 Categorical 換成 Normal，PPO 主體
      一行不改。這是本系列從沒碰過的軸（前面動作全是離散的）。
  (2) 反面教材：誠實示範「單步 TD δ 的極限」。本檔在 Pendulum 上「學不起來」，
      而且這不是 bug、也不是參數沒調好（lr / 正規化 / reward 縮放都試過，
      見文末）——是演算法本身不夠用。它正好引出下一個階段：GAE。

  ⚠️ 預期結果：平均 return 大致停在 -1200 ~ -1600（和隨機策略差不多），
     不會明顯往 0 爬。看到它「沒進步」就對了，那是這個階段要教的事。

【一句話總結這個階段】

  階段 1~5 的動作全是「離散」的——CartPole 只有「左推 / 右推」兩個選擇，
  policy 用 Categorical（softmax）輸出每個動作的機率。

  但真實世界很多控制是「連續」的：施多大的力、轉幾度、踩多深的油門。
  這時 policy 不能再列舉動作，得改成輸出一個「連續分布」，從中取樣。
  最常用的是高斯分布 Normal(mean, std)：
    actor 輸出平均 mean（這個狀態下「大概該施多少力」），
    再配一個標準差 std 決定探索幅度，動作 = 從 Normal 取樣。

  ★ 關鍵洞見：PPO 的核心（ratio + clip + 同批 K 次複用）跟動作分布
    「完全無關」。ratio = π_new/π_old 只看 log_prob，不在乎 π 是
    Categorical 還是 Normal。所以本檔相對 ppo_cartpole.py 的 diff
    乾淨到只剩「換分布」這一件事——這正是要演示的重點。

【相比 ppo_cartpole.py 的 diff（只有這些，PPO 主體一行不改）】

  1. Actor 網路：輸出 logits(2) → 改輸出連續高斯的 mean(1) + log_std。
     （GaussianActorNetwork，見下方；Critic 完全不變，只是輸入維度 3）
  2. 動作分布：Categorical(logits=...) → Normal(mean, std)。
     choose_action 從 Normal 取樣（而非 np.random.choice）。
  3. log_prob：連續動作要對「動作維度」加總 → .sum(axis=-1)。
     （Pendulum 動作只有 1 維，加不加總數值一樣，但寫成通用形式）
  4. 動作裁剪：環境力矩限制在 [-2, 2]，送進環境前 clip；
     但 log_prob 用「未裁剪的原始取樣動作」算（見 store/update 註解），
     才能讓 ratio 的 important sampling 維持正確。

  update() 裡的 ratio、surr1/surr2、clip、TD target、advantage、critic
  ——和 ppo_cartpole.py 逐字相同。把兩個檔案對放，diff 只落在「分布」。

【Pendulum-v1 這個環境（和 CartPole 的差異）】

  觀測 (3 維)：[cos θ, sin θ, 角速度]  ← 倒立擺的角度與轉速
  動作 (1 維)：施加的力矩，連續值 ∈ [-2, 2]
  reward    ：負的成本 = -(θ² + 0.1·θ̇² + 0.001·力矩²)
              → 桿子越接近「正上方且靜止」，reward 越接近 0（最好）；
                越歪、轉越快、用越大力，reward 越負。
  ⚠️ 沒有「提早終止」：每條 episode 固定跑滿 200 步才 truncated，
     terminated 永遠是 False。
     → 意味 TD target 的 V(s') 永遠 bootstrap（從不歸零）。
       這和 CartPole「桿子倒 → terminated → V(s')=0」不同；
       但程式碼完全不用改——terminateds 全 False，自然就一路 bootstrap。

  Pendulum 沒有官方「解決」門檻（reward 恆為負）。經驗值：
    隨機策略約 -1200 ~ -1600；學會把桿子甩上去並穩住約 -150 ~ -250。
    本檔（極簡 PPO）會停在隨機水準——原因見下。

────────────────────────────────────────────────────────────
【為什麼極簡 PPO 在 Pendulum 學不起來？（本階段的核心反例）】

  Pendulum 是「swing-up」：桿子一開始垂在下面，馬達力矩 ≪ 重力，沒辦法
  一步硬抬上去。正確解法是「先來回擺、累積動量，幾十步後才甩到頂端」。
  → 這需要把「現在多施點力」和「幾十步後桿子終於立起來」這個遙遠的好結果
    連起來，也就是「長程信用分配」。

  但本檔的 advantage 用的是「單步 TD δ = r + γV(s') − V(s)」：
    它只看「下一步」的價值差。在 swing-up 裡，擺動過程每一步的即時 reward
    都很負、單步價值差又被未訓練好的 critic 噪聲淹沒，δ 幾乎傳遞不到
    「該為了幾十步後的成功而現在出力」這個訊號。actor 收到的梯度等於是噪聲，
    於是原地打轉。CartPole 能成功是因為它 dense +1、且「撐住」這件事每一步
    都能即時反映，單步 δ 就夠用；Pendulum 不行。

  ⚠️ 這不是調參能解決的（已實測）：
       - lr 掃 0.0003 / 0.001 / 0.003 → 沒有一個學得起來，甚至更糟（策略崩）
       - 加 advantage 正規化、加 reward 縮放 → 仍停在 -1370 上下
     證明瓶頸在「advantage 估計法」本身，不在超參。

  ★ 這正好引出下一個工具：GAE（Generalized Advantage Estimation）。
    GAE 用 λ 把「單步 TD δ」和「多步 / Monte Carlo」之間做加權折衷：
      Â_t = Σ (γλ)^l · δ_{t+l}
    λ=0 退化回單步 TD（就是本檔，高 bias）；λ=1 接近 MC（高 variance）；
    λ≈0.95 兩者兼顧，能把遙遠的成功訊號沿著軌跡一路傳回來——這就是
    Pendulum / 機器人控制能訓得起來的關鍵。本系列刻意把它留到下一階段，
    讓「為什麼需要 GAE」由本檔的失敗親自說明，而不是憑空塞一條公式。

────────────────────────────────────────────────────────────
【連續動作的 log_prob 為什麼長這樣？】

  離散：log π(a|s) = log( softmax(logits)[a] )           ← 一個機率值取 log
  連續：log π(a|s) = Normal(mean, std).log_prob(a)
                  = -½·((a-mean)/std)² - log(std) - ½·log(2π)
        （這是高斯機率「密度」的 log；對多維動作各維相加 → .sum(-1)）

  直覺一樣：動作 a 離 mean 越近、std 越合適 → log_prob 越大；
  policy gradient 就是去調 mean / std，讓「advantage 為正」的動作
  log_prob 上升、為負的下降——和離散版的精神完全一致。

  std 怎麼來？本檔用「與狀態無關的可學習 log_std」（一個 nn.Parameter，
  CleanRL / SpinningUp 的標準最小做法）：訓練初期 std 大 → 多探索，
  學會後 std 自動縮小 → 動作越來越確定。log_std 也吃 actor 的梯度。
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
#  （這是相對 ppo_cartpole.py 真正改動的地方）
# ─────────────────────────────────────────────────────────

class GaussianActorNetwork(nn.Module):
    """
    連續動作 Actor：輸入狀態，輸出高斯分布的「平均 mean」。
    架構：輸入(3) → 隱藏層(64, ReLU) → 輸出(1, mean，線性)

    和離散版 PolicyNetwork 的差異：
      離散：輸出 2 個 logits → Categorical → 機率（加總為 1）
      連續：輸出 1 個 mean  → 配 log_std → Normal(mean, std)

    log_std：與狀態無關的可學習參數（不經網路，直接是一個 nn.Parameter）。
      這是最小可行做法：訓練初期 std≈1 鼓勵探索，學會後自動縮小。
      它在 actor.parameters() 裡，所以一樣吃 actor_opt 的梯度。
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
#  PPO Agent（與 ppo_cartpole.py 結構相同，僅分布相關處改動）
# ─────────────────────────────────────────────────────────

class PPOAgent:
    def __init__(self, action_low, action_high):
        self.actor  = GaussianActorNetwork()
        self.critic = CriticNetwork(input_dim=3)   # Pendulum 觀測是 3 維

        self.gamma = 0.99
        # 刻意對齊 ppo_cartpole.py：actor_lr=0.001、critic_lr=0.005，完全相同。
        # 這是本檔的重點之一——相對離散版「真的只改了動作分布」，連 lr 都沒動。
        # 於是後面看到 Pendulum 學不起來，就能確定問題不在「沒調好參數」，而在
        # 演算法本身（單步 TD δ 撐不起 swing-up 的長程信用分配）。
        # （實測：另外掃過 lr、advantage 正規化、reward 縮放都救不起來，見檔頭說明。）
        self.actor_lr  = 0.001
        self.critic_lr = 0.005

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=self.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # ── PPO 核心超參（與離散版相同）──
        self.clip_eps = 0.2   # ε：ratio 被夾在 [1-ε, 1+ε]
        self.k_epochs = 4     # 同一批資料整批重複更新幾次（full-batch，不切 minibatch）

        # 動作邊界（送進環境前用來 clip）
        self.action_low  = action_low
        self.action_high = action_high

        # buffer：和 ppo_cartpole.py 相同
        self._states      = []
        self._actions     = []
        self._rewards     = []
        self._next_states = []
        self._terminateds = []
        self._dones       = []

    @torch.no_grad()
    def choose_action(self, state):
        """
        從高斯分布取樣連續動作（離散版是 np.random.choice，這裡改 Normal.sample）。

        回傳兩個東西：
          raw_action  ：分布原始取樣值，存進 buffer 給 log_prob 用（不裁剪）
          env_action  ：clip 到 [low, high] 後送進環境
        為什麼分開？若拿「裁剪後」的動作回去算 log_prob，邊界處的密度會失真，
        ratio 的 important sampling 就不準了。標準做法：log_prob 用原始取樣值，
        裁剪只發生在「丟給環境」這一步。
        """
        x = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        mean, std  = self.actor(x)
        raw_action = Normal(mean, std).sample()[0].numpy()        # (action_dim,) 未裁剪
        env_action = np.clip(raw_action, self.action_low, self.action_high)
        return raw_action, env_action

    def store(self, state, raw_action, reward, next_state, terminated, done):
        # ⚠️ 存「未裁剪」的 raw_action（見 choose_action 說明）
        self._states.append(state)
        self._actions.append(raw_action)
        self._rewards.append(reward)
        self._next_states.append(next_state)
        self._terminateds.append(terminated)
        self._dones.append(done)

    def update(self):
        states      = torch.as_tensor(np.array(self._states),      dtype=torch.float32)  # (T,3)
        actions     = torch.as_tensor(np.array(self._actions),     dtype=torch.float32)  # (T,1) 連續！
        rewards     = torch.as_tensor(np.array(self._rewards),     dtype=torch.float32)  # (T,)
        next_states = torch.as_tensor(np.array(self._next_states), dtype=torch.float32)  # (T,3)
        terminateds = torch.as_tensor(np.array(self._terminateds), dtype=torch.bool)     # (T,) 此環境恆 False
        T = len(self._rewards)
        N = int(np.sum(self._dones))   # 完整軌跡數，actor loss 除以它 → 與離散版一致

        # ══════════════════════════════════════════════════════════
        #  Phase 1：更新前，用「舊策略 / 舊 critic」算好並凍結
        #           （和 ppo_cartpole.py 逐字相同，僅分布從 Categorical → Normal）
        # ══════════════════════════════════════════════════════════
        with torch.no_grad():
            # (a) old_log_prob = log π_old(a|s)：此刻的 actor 就是 π_old
            #     ★ 唯一的分布改動：Categorical(logits) → Normal(mean, std)
            #       連續動作對「動作維度」加總 log_prob → .sum(axis=-1)
            old_mean, old_std = self.actor(states)                            # (T,1),(1,)
            old_log_probs = Normal(old_mean, old_std).log_prob(actions).sum(axis=-1)  # (T,)

            # (b) TD target = r + γV(s')，只有真終止才把 V(s') 歸零
            #     （Pendulum 永不 terminated → 永遠 bootstrap，與 TD 版處理相同）
            next_values = self.critic(next_states)                   # (T,)  V(s_{t+1})
            next_values[terminateds] = 0.0                           # 此環境恆 False，等同沒作用
            td_targets = rewards + self.gamma * next_values          # (T,)  y_t

            # (c) advantage = TD target - V_old(s)，單步 TD δ（不用 GAE）
            old_values = self.critic(states)                         # (T,)  V(s_t)
            advantages = td_targets - old_values                     # (T,)  Â_t

        # ══════════════════════════════════════════════════════════
        #  Phase 2：同一批資料（整批），重複做 K 次完整 update
        #           ── 從這裡到結尾，和 ppo_cartpole.py 唯一的差別就是
        #              「new_logp 用 Normal 而非 Categorical」這一行。
        #              ratio / clip / surrogate / critic 全部一字不差。
        # ══════════════════════════════════════════════════════════
        for _ in range(self.k_epochs):
            # ── 更新 Actor：clipped surrogate（整批）──────────
            # log π_θ(a|s)：ratio 的分子，用「當前」θ 算（★ 分布改動的唯一一行）
            mean, std = self.actor(states)                            # (T,1),(1,)
            new_logp  = Normal(mean, std).log_prob(actions).sum(axis=-1)  # (T,)

            # r_t(θ) = exp(log π_θ − log π_θ_old)，做 important sampling
            ratio = torch.exp(new_logp - old_log_probs)              # r_t(θ) = π_new/π_old
            surr1 = ratio * advantages                               # r_t(θ)·Â_t
            surr2 = torch.clamp(ratio, 1 - self.clip_eps,
                                       1 + self.clip_eps) * advantages  # clip(r_t,1−ε,1+ε)·Â_t

            # L^CLIP = E[min(surr1, surr2)]，最大化它 → loss 取負；/N 對齊離散版。
            actor_loss = -torch.min(surr1, surr2).sum() / N          # −L^CLIP(θ)
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # ── 更新 Critic：value MSE（target 已凍結，整批）──
            v = self.critic(states)                                  # (T,)  V_φ(s_t)
            critic_loss = F.mse_loss(v, td_targets)                  # (V_φ(s_t) − y_t)²
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

        # 清空 buffer
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

    # 對齊離散版：batch=4、共 250 次更新、總計 1000 episodes。
    # Pendulum 每條 episode 固定 200 步，故每次更新看 4×200=800 步資料。
    BATCH_EPISODES = 4     # 每收集 4 條軌跡才更新一次（batch size，與離散版相同）
    NUM_UPDATES    = 250   # 總共要更新幾次（與離散版相同）
    EPISODES       = BATCH_EPISODES * NUM_UPDATES
    scores         = []

    print("=" * 60)
    print("  RL Hello World 5b — PPO 連續動作版 on Pendulum [PyTorch]")
    print("=" * 60)
    print(f"\nclip ε = {agent.clip_eps}   K_epochs = {agent.k_epochs}   "
          f"(full-batch，不切 minibatch)")
    print("核心   ：和 ppo_cartpole.py 完全相同（ratio + clip + 同批 K 次複用）")
    print("唯一改動：動作分布 Categorical → Normal(mean, std)（連續動作）")
    print("⚠️ 預期：學不起來（return 停在隨機水準 -1200~-1600）。")
    print("   這是刻意的反例——單步 TD δ 撐不起 swing-up，引出下一階段 GAE。")
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

        if (episode + 1) % 50 == 0:
            avg_score = np.mean(scores[-50:])
            print(f"Episode {episode+1:4d} | 近50回合平均 return: {avg_score:8.1f}")

    env.close()

    # ── 展示（用確定性的 mean 動作，不取樣）──
    print("\n" + "=" * 60)
    print("  訓練完成！跑 5 次展示（用 mean 動作）")
    print("=" * 60)
    env = gym.make("Pendulum-v1")
    for trial in range(5):
        state, _ = env.reset()
        total_reward = 0
        while True:
            mean_action = agent.actor.predict_mean(state)
            env_action  = np.clip(mean_action, action_low, action_high)
            state, reward, terminated, truncated, _ = env.step(env_action)
            total_reward += reward
            if terminated or truncated:
                break
        print(f"  Trial {trial+1}：return = {total_reward:8.1f}")
    env.close()

    # ── 對比總結：離散 PPO vs 連續 PPO ──
    print("\n" + "=" * 60)
    print("  離散 PPO (CartPole) vs 連續 PPO (Pendulum)")
    print("=" * 60)
    rows = [
        ("動作空間",     "離散（左/右）",            "連續（力矩 ∈ [-2,2]）"),
        ("Actor 輸出",   "logits(2)",                "mean(1) + log_std"),
        ("動作分布",     "Categorical(logits)",      "Normal(mean, std)"),
        ("取樣動作",     "np.random.choice",         "Normal.sample + clip"),
        ("PPO 核心",     "ratio + clip + K 複用",    "完全相同（一行不改）"),
    ]
    print(f"  {'':12s} {'離散 PPO':28s} {'連續 PPO':28s}")
    print("  " + "-" * 70)
    for label, a, p in rows:
        print(f"  {label:12s} {a:28s} {p:28s}")
    print("\n核心洞見（兩個）：")
    print("  1. 正面：PPO 的 ratio + clip 與動作分布無關——換成高斯，主體一行不改。")
    print("           連續控制（機器人、MuJoCo）用的就是這種高斯 policy + PPO。")
    print("  2. 反面：上面 return 停在隨機水準 ≈ -1400，沒往 0 爬。")
    print("           單步 TD δ 撐不起 Pendulum 的 swing-up（長程信用分配）。")
    print("           → 下一階段 GAE 用 λ 把多步訊號傳回來，正是為了解決這個。")

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
