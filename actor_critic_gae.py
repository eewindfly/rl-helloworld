"""
RL Hello World 4c — Actor-Critic (A2C) GAE 版 on CartPole
=========================================================
和 actor_critic_td.py 的唯一「概念」差異：把單步 TD δ 換成 GAE。

【一句話：GAE 是把階段 4（MC）和階段 4b（TD）統一起來的那根旋鈕】

  你已經做過 advantage 的兩個極端：
    階段 4  AC(MC)：A_t = G_t − V(s_t)        ← 用「整段 return」估，無 bias、方差高
    階段 4b AC(TD)：A_t = δ_t = r+γV(s')−V(s) ← 用「單步」估，  有 bias、方差低

  GAE（Generalized Advantage Estimation, Schulman 2015）用一個 λ∈[0,1]
  在這兩者之間連續插值：

    Â_t^GAE = Σ_{l≥0} (γλ)^l · δ_{t+l}
            = δ_t + γλ·δ_{t+1} + (γλ)²·δ_{t+2} + ...

    λ = 0  →  Â_t = δ_t               （退化成階段 4b 的單步 TD）
    λ = 1  →  Â_t = Σ γ^l r − V(s_t)  （退化成階段 4 的 MC，無 bias）
    λ ≈ 0.95（本檔）→ 兩者折衷：保留多步訊號、又不像 MC 那麼吵

  直覺：δ_t 只看下一步；GAE 把「後面好幾步的 δ」用 (γλ)^l 衰減後加總，
  於是「現在這個動作」能沾到「幾步之後才兌現的好結果」——這正是
  單步 TD 做不到、而長程任務（如下個階段的 Pendulum swing-up）需要的。

【GAE 的 critic target：λ-return（維持本系列的不變式）】

  本系列從頭到尾有個不變式：critic target = advantage + V(s)。
    階段 4  MC：target = (G_t − V) + V = G_t
    階段 4b TD：target = (δ   − 0) ... 即 r+γV(s')  （= advantage + V）
  GAE 也照辦：
    target = Â_t^GAE + V(s_t)   ← 這東西就叫「λ-return」，是 TD(λ) 的目標。
  所以「換成 GAE」這一個動作，同時決定了 advantage 與 critic target，
  和階段 4 / 4b 的模式一模一樣——依然是「只改一個東西：advantage 的估法」。

【程式碼改動（相比 actor_critic_td.py）】

  1. 多一個超參 gae_lambda（λ=0.95）。
  2. update() 裡：把「advantage = td_target − V」這一行，換成
     一段「反向掃描軌跡、累積 GAE」的計算（見 compute_gae 註解），
     critic target 改用 λ-return（= advantage + V）。
  3. 其餘（actor loss、batch 更新時機、網路、lr）全部和 4b 相同。

  ⚠️ 反向掃描要在「每條 episode 邊界」重置累積值（done 為界），
     不能讓上一條軌跡的 advantage 漏進下一條。

────────────────────────────────────────────────────────────
【為什麼 CartPole 上看不太出 GAE 的威力？（誠實說在前面）】

  CartPole 是 dense +1、且單步 δ 本來就夠用的環境，所以這裡 GAE 的表現
  和階段 4b 差不多（一樣能解到 ≥195）。GAE 真正的價值要到「長程信用分配」
  的任務才看得出來——下個階段把同一套 GAE 接到 PPO、丟上 Pendulum
  (swing-up) 時，你會看到 λ=0（單步 TD）學不起來、λ=0.95（GAE）才學得起來。
  本檔的任務是先把 GAE 這個工具「乾淨地」介紹清楚，威力留到那時驗證。
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
from utils import demo
from pg_cartpole import PolicyNetwork as ActorNetwork
from actor_critic import CriticNetwork   # Critic 架構完全相同，直接複用


# ─────────────────────────────────────────────────────────
#  Actor-Critic GAE Agent
# ─────────────────────────────────────────────────────────

class ACGAEAgent:
    def __init__(self):
        self.actor  = ActorNetwork()
        self.critic = CriticNetwork()

        self.gamma      = 0.99
        self.gae_lambda = 0.95   # ★ 唯一的新超參：λ=0→單步TD(4b)，λ=1→MC(4),λ=0.95→折衷
        # lr 沿用 4b（actor 0.001 / critic 0.005），不另調——讓 diff 純剩 GAE。
        self.actor_lr  = 0.001
        self.critic_lr = 0.005

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=self.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # buffer 和 4b 完全相同
        self._states      = []
        self._actions     = []
        self._rewards     = []
        self._next_states = []
        self._terminateds = []
        self._dones       = []

    def choose_action(self, state):
        probs = self.actor.predict_probs(state)
        return np.random.choice(len(probs), p=probs)

    def store(self, state, action, reward, next_state, terminated, done):
        self._states.append(state)
        self._actions.append(action)
        self._rewards.append(reward)
        self._next_states.append(next_state)
        self._terminateds.append(terminated)
        self._dones.append(done)

    def compute_gae(self, deltas, dones):
        """
        反向掃描軌跡，把單步 δ 累積成 GAE：
            Â_t = δ_t + γλ·Â_{t+1}
        這是 Â_t = Σ (γλ)^l δ_{t+l} 的等價遞迴寫法（從後往前算最省事）。

        ⚠️ dones[t]=True 表示 t 是某條 episode 的最後一步 → 在它「之後」
           沒有同一條軌跡的未來，所以把累積值 last 歸零再算這一步。
           （δ_t 本身已透過 next_value 處理好終止/截斷的 bootstrap，
             這裡只需確保 GAE 的累積不跨越 episode 邊界。）
        """
        T = len(deltas)
        advantages = torch.zeros(T)
        last = 0.0
        for t in reversed(range(T)):
            if dones[t]:
                last = 0.0                              # episode 邊界：重置累積
            last = deltas[t] + self.gamma * self.gae_lambda * last
            advantages[t] = last
        return advantages

    def update(self):
        states      = torch.as_tensor(np.array(self._states),      dtype=torch.float32)  # (T,4)
        actions     = torch.as_tensor(np.array(self._actions),     dtype=torch.long)     # (T,)
        rewards     = torch.as_tensor(np.array(self._rewards),     dtype=torch.float32)  # (T,)
        next_states = torch.as_tensor(np.array(self._next_states), dtype=torch.float32)  # (T,4)
        terminateds = torch.as_tensor(np.array(self._terminateds), dtype=torch.bool)     # (T,)
        N = int(np.sum(self._dones))   # 完整軌跡數（語意同 4b）

        # ── 步驟 1：算每步 TD 殘差 δ，再反向累積成 GAE（都是「目標」，no_grad）──
        #
        #   δ_t = r_t + γV(s') − V(s_t)   ← 和 4b 的單步 δ 完全相同
        #   Â_t = Σ (γλ)^l δ_{t+l}        ← 4b 只取 l=0 那一項；GAE 把後面都加進來
        #   λ-return = Â_t + V(s_t)        ← critic target（維持 target=adv+V 不變式）
        #
        #   V(s') 只在真終止 terminated 時歸零；truncated 照常 bootstrap（同 4b）。
        with torch.no_grad():
            values      = self.critic(states)                    # (T,) V(s_t)
            next_values = self.critic(next_states)               # (T,) V(s_{t+1})
            next_values[terminateds] = 0.0
            deltas      = rewards + self.gamma * next_values - values   # (T,) δ_t
            advantages  = self.compute_gae(deltas, self._dones)         # (T,) Â_t^GAE
            returns     = advantages + values                           # (T,) λ-return（critic target）

        # ── 步驟 2：更新 Critic（讓 V(s) → λ-return，MSE）──
        #   對比 4b：唯一差別是 target 從「單步 td_target」換成「λ-return」。
        values_pred = self.critic(states)                        # (T,) 留梯度
        critic_loss = F.mse_loss(values_pred, returns)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ── 步驟 3：更新 Actor（和 4b 一字不差，只是 advantage 換成 GAE）──
        logits     = self.actor(states)            # (T, 2)
        log_probs  = Categorical(logits=logits).log_prob(actions)  # (T,)
        actor_loss = -(log_probs * advantages).sum() / N
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # 清空
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
    env   = gym.make("CartPole-v1")
    agent = ACGAEAgent()

    BATCH_EPISODES = 4     # 每收集幾條軌跡才更新一次（和 4b 相同）
    NUM_UPDATES    = 250   # 總共更新幾次（和 4b 相同）
    EPISODES       = BATCH_EPISODES * NUM_UPDATES
    scores         = []

    print("=" * 60)
    print("  RL Hello World 4c — Actor-Critic (A2C) GAE 版 [PyTorch]")
    print("=" * 60)
    print(f"\nGAE Advantage：Â_t = Σ (γλ)^l δ_{{t+l}}   (λ = {agent.gae_lambda})")
    print("對比 4b（TD）：Â_t = δ_t        （= GAE 在 λ=0）")
    print("對比 4 （MC）：Â_t = G_t - V(s) （≈ GAE 在 λ=1）")
    print("差異        ：只改 advantage 的估法（多步加權），其餘同 4b")
    print("目標        ：近 50 回合平均分 ≥ 195 = 解決！")
    print(f"\n開始訓練 {EPISODES} 個 episodes...\n")

    for episode in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0

        while True:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store(state, action, reward, next_state, terminated, done)
            total_reward += reward
            state = next_state
            if done:
                break

        scores.append(total_reward)

        if (episode + 1) % BATCH_EPISODES == 0:
            agent.update()

        if (episode + 1) % 50 == 0:
            avg_score = np.mean(scores[-50:])
            solved = "✓ 解決！" if avg_score >= 195 else ""
            print(f"Episode {episode+1:4d} | "
                  f"近50回合平均分: {avg_score:6.1f}  {solved}")

    env.close()

    # ── 展示 ──
    print("\n" + "=" * 60)
    print("  訓練完成！跑 5 次展示")
    print("=" * 60)
    env = gym.make("CartPole-v1")
    for trial in range(5):
        state, _ = env.reset()
        steps = 0
        while True:
            probs  = agent.actor.predict_probs(state)
            action = int(np.argmax(probs))
            state, _, terminated, truncated, _ = env.step(action)
            steps += 1
            if terminated or truncated:
                break
        result = "✓ 撐住了！" if steps >= 195 else f"倒了（{steps} 步）"
        print(f"  Trial {trial+1}：{steps} 步  {result}")
    env.close()

    # ── 對比總結：TD vs GAE ──
    print("\n" + "=" * 60)
    print("  A2C TD vs GAE")
    print("=" * 60)
    rows = [
        ("Advantage",     "δ_t（單步）",               "Σ(γλ)^l·δ（多步加權）"),
        ("是 GAE 的",     "λ=0 特例",                  "一般式（λ=0.95）"),
        ("Critic target", "r + γV(s')",                "λ-return（= Â + V）"),
        ("長程信用分配",   "弱（只看一步）",            "強（多步傳遞）"),
        ("CartPole 表現",  "夠用",                      "差不多（要難任務才見差）"),
    ]
    print(f"  {'':14s} {'TD (4b)':28s} {'GAE (4c)':28s}")
    print("  " + "-" * 74)
    for label, td, gae in rows:
        print(f"  {label:14s} {td:28s} {gae:28s}")
    print("\n核心洞見：")
    print("  GAE 用 λ 把 MC（階段4）和 TD（階段4b）連成一條光譜，λ=0.95 取折衷。")
    print("  真 PPO（下一階段）標配的就是 GAE——這是接進 PPO 的正確一塊拼圖。")

    return agent


if __name__ == "__main__":
    agent = train()
    demo(lambda state: int(np.argmax(agent.actor.predict_probs(state))))
