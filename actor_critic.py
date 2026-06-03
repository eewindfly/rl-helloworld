"""
RL Hello World 4 — Actor-Critic (A2C) on CartPole
==================================================
從 Policy Gradient（REINFORCE）進化到 Actor-Critic

【REINFORCE 的問題】

  REINFORCE 用 G_t（整段的 Monte Carlo return）當學習信號。
  問題：同一個動作在不同 episode 中，G_t 可能差很多，
        梯度估計的「方差」很大，訓練不穩定。

  根本原因：G_t 包含太多不確定性（後面還沒發生的隨機事件）。

【Actor-Critic 的解法：Baseline + Advantage】

  核心思路：
    不看動作帶來多少絕對回報，而是看它「比平均好多少」。
    「平均」= V(s)：從狀態 s 開始，按當前 policy 能拿到的期望回報。

    Advantage A(s, a) = Q(s, a) - V(s)
                      ≈ G_t - V(s)

    A > 0 → 這個動作比平均好，提高機率
    A < 0 → 這個動作比平均差，降低機率
    A ≈ 0 → 差不多，幾乎不更新

  效果：Advantage 的方差遠小於 G_t。

【為什麼叫 Actor-Critic？】

  Actor  = policy π(a|s)：決定要做什麼動作
  Critic = value  V(s)  ：評估當前狀態好不好，給 Actor 回饋

  類比：
    Actor  = 演員，決定怎麼演
    Critic = 導演，告訴演員哪段演得好、哪段演得差（比平均好多少）

【和 REINFORCE 的程式碼差異（只有兩處）】

  1. 多一個 CriticNetwork，用來估 V(s)
  2. 更新 Actor 時，把 G_t 換成 Advantage = G_t - V(s)

  其餘結構（訓練迴圈、反向傳播邏輯）和 pg_cartpole.py 完全相同。

【MC vs TD：兩種估計 Q 的方式】

  Advantage = Q(s,a) - V(s)，Q 不直接 train，用估計替代：

  本實作用 MC 版本。

                    MC                        TD
  ──────────────────────────────────────────────────────────
  Q 估計            G_t                       r_t + γV(s_{t+1})
  V 的訓練 target   G_t                       r_t + γV(s_{t+1})
  Advantage         G_t - V(s_t)              r_t + γV(s_{t+1}) - V(s_t)
  方差              高                        低
  Bias              無                        有（V 不準就偏）
  適用任務          有終點的任務              有終點或無終點皆可
  Actor 更新時機    N 個 episode 結束後       N 個 episode 結束後
  Critic 更新時機   episode 結束後            理論上每步可更新，實務通常同 Actor

  兩套各自一致：Q 的估計方式決定了 V 的訓練 target，是同一個選擇。
"""

import numpy as np
import gymnasium as gym
from utils import demo
from pg_cartpole import PolicyNetwork as ActorNetwork   # Actor 和 REINFORCE 的 policy 網路完全相同


# ─────────────────────────────────────────────────────────
#  2. Critic 網路（新增！）
# ─────────────────────────────────────────────────────────

class CriticNetwork:
    """
    輸入狀態，輸出一個純量 V(s)（預期的總回報）
    架構：4 → 64（ReLU）→ 1（線性）

    和 Actor 的差異：
      Actor  → 輸出 2 個機率（softmax，加總為 1）
      Critic → 輸出 1 個數值（無激活函式，任意實數）

    Critic 的訓練目標：
      讓 V(s) 盡量接近 G_t（Monte Carlo return）
      Loss = MSE = mean( (V(s) - G_t)^2 )
    """
    def __init__(self, input_dim=4, hidden_dim=64):
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(1)

    def forward(self, x):
        self.x = x
        self.h = np.maximum(0, x @ self.W1 + self.b1)
        self.v = self.h @ self.W2 + self.b2   # (T, 1)，線性輸出
        return self.v.squeeze(-1)              # (T,)

    def backward(self, grad_v, lr):
        """
        grad_v = dLoss/dV = (V(s) - G_t) / T，形狀 (T,)
        """
        grad_v = grad_v.reshape(-1, 1)         # (T, 1)
        dW2 = self.h.T @ grad_v
        db2 = grad_v.sum(axis=0)
        grad_h = grad_v @ self.W2.T
        grad_h[self.h <= 0] = 0
        dW1 = self.x.T @ grad_h
        db1 = grad_h.sum(axis=0)
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def predict_value(self, state):
        # forward 回傳 shape (1,) 的陣列，要先取出元素再轉純量
        # （直接 float(陣列) 在新版 numpy 會丟 TypeError）
        return float(self.forward(state.reshape(1, -1))[0])


# ─────────────────────────────────────────────────────────
#  3. Actor-Critic Agent
# ─────────────────────────────────────────────────────────

class ACAgent:
    def __init__(self):
        self.actor  = ActorNetwork()
        self.critic = CriticNetwork()

        # 超參數
        self.gamma     = 0.99
        self.actor_lr  = 0.0005
        self.critic_lr = 0.001   # Critic 通常用較高學習率，讓 V(s) 快點準確

        # 存放多條軌跡（和 PG 相同）
        self.trajectories = []      # list of (states, actions, rewards)
        self._cur_states  = []      # 當前 episode 暫存
        self._cur_actions = []
        self._cur_rewards = []

    def choose_action(self, state):
        probs = self.actor.predict_probs(state)
        return np.random.choice(len(probs), p=probs)

    def store(self, state, action, reward):
        self._cur_states.append(state)
        self._cur_actions.append(action)
        self._cur_rewards.append(reward)

    def end_episode(self, terminated, last_next_state):
        """
        Episode 結束，把這條軌跡存起來。

        多存兩樣東西，給「截斷 bootstrap」用（見 compute_returns）：
          terminated      ：是不是真終止（桿子倒了）。
                            truncated（撐到時間上限被截斷）時為 False。
          last_next_state ：這條軌跡最後一步的 s'，truncated 時要用它 bootstrap。
        """
        self.trajectories.append((
            list(self._cur_states),
            list(self._cur_actions),
            list(self._cur_rewards),
            terminated,
            last_next_state,
        ))
        self._cur_states  = []
        self._cur_actions = []
        self._cur_rewards = []

    def compute_returns(self, rewards, terminated, last_next_state):
        """
        Monte Carlo return：從每一步開始的累積折扣獎勵 G_t

        ⚠️ 累加起點 acc 不能無腦設 0！
           - terminated（桿子倒了）：之後真的沒有未來獎勵 → acc 從 0 起算，正確。
           - truncated（撐到 500 步上限被截斷）：桿子還立著，後面本來還有一大段
             獎勵，只是被時間上限切掉。若 acc 從 0 起算，會把接近上限的那幾步
             G_t 系統性低估（少算截斷後的尾巴），剛好打到「滿分軌跡」。
             正確做法：用 critic 估的 V(last_next_state) 當尾巴 bootstrap。
        """
        T = len(rewards)
        G = np.zeros(T)
        acc = 0.0 if terminated else self.critic.predict_value(last_next_state)
        for t in reversed(range(T)):
            acc = rewards[t] + self.gamma * acc
            G[t] = acc
        return G

    def update(self):
        """
        用收集到的 N 條軌跡同時更新 Actor 和 Critic（和 PG 相同結構）

        流程：
          1. 每條軌跡各自算 G_t，合併成一個大 batch
          2. Critic 前向：估 V(s) for 所有狀態
          3. Advantage = G_t - V(s)
          4. 更新 Critic：MSE loss，讓 V(s) → G_t
          5. 更新 Actor ：用 Advantage 替換 REINFORCE 的 G_t，除以 N
        """
        N = len(self.trajectories)

        # 步驟 1：每條軌跡各自算 return，再合併（和 PG 相同）
        all_states   = []
        all_actions  = []
        all_returns  = []

        for states, actions, rewards, terminated, last_next_state in self.trajectories:
            returns = self.compute_returns(rewards, terminated, last_next_state)
            all_states.extend(states)
            all_actions.extend(actions)
            all_returns.extend(returns)

        states  = np.array(all_states)
        actions = np.array(all_actions)
        returns = np.array(all_returns)
        T = len(states)

        # 步驟 2：Critic 估 V(s)
        values = self.critic.forward(states)   # (T,)

        # 步驟 3：Advantage
        #   REINFORCE：normalize(G_t) → 用全局 mean 當 baseline
        #   Actor-Critic：G_t - V(s)  → 用每個狀態的期望值當 baseline（更精確）
        advantage = returns - values   # (T,)

        # 步驟 4：更新 Critic
        #   Loss = mean( (V(s) - G_t)^2 )，dL/dV = (V(s) - G_t) / T
        grad_v = (values - returns) / T
        self.critic.backward(grad_v, self.critic_lr)

        # 步驟 5：更新 Actor
        #   ∇J = Σ_t ∇ log π(a_t | s_t) × A_t
        #   除以 N（對應公式的 1/N，和 PG 相同）
        probs = self.actor.forward(states)          # (T, 2)
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(T), actions] = 1.0
        grad_logits = -(one_hot - probs) * advantage.reshape(-1, 1)
        grad_logits /= N    # 對應公式的 1/N，和 PG 相同
        self.actor.backward(grad_logits, self.actor_lr)

        # 清空（on-policy：跑完就丟）
        self.trajectories = []


# ─────────────────────────────────────────────────────────
#  4. 訓練迴圈
# ─────────────────────────────────────────────────────────

def train():
    env   = gym.make("CartPole-v1")
    agent = ACAgent()

    EPISODES  = 1000
    N_UPDATES = 4     # 每收集幾條軌跡才更新一次（和 PG 相同）
    scores    = []

    print("=" * 60)
    print("  RL Hello World 4 — Actor-Critic (A2C)")
    print("=" * 60)
    print("\n核心概念：Advantage = G_t - V(s)，Critic 降低梯度方差")
    print("兩個網路：Actor（學機率）+ Critic（學狀態價值）")
    print(f"更新時機：每收集 {N_UPDATES} 條軌跡後更新（和 PG 相同）")
    print("目標    ：近 50 回合平均分 ≥ 195 = 解決！")
    print(f"\n開始訓練 {EPISODES} 個 episodes...\n")

    for episode in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0

        while True:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store(state, action, reward)
            total_reward += reward
            state = next_state

            if done:
                break

        # 傳 terminated（區分真終止 vs 截斷）和最後的 next_state（截斷時 bootstrap 用）
        agent.end_episode(terminated, next_state)
        scores.append(total_reward)

        # 每收集 N_UPDATES 條軌跡才更新一次（和 PG 相同）
        if (episode + 1) % N_UPDATES == 0:
            agent.update()

        if (episode + 1) % 50 == 0:
            avg_score = np.mean(scores[-50:])
            solved = "✓ 解決！" if avg_score >= 195 else ""
            print(f"Episode {episode+1:4d} | "
                  f"近50回合平均分: {avg_score:6.1f}  {solved}")

    env.close()

    # ─────────────────────────────────────────────────────
    #  5. 展示訓練完的 Agent
    # ─────────────────────────────────────────────────────
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

    # ─────────────────────────────────────────────────────
    #  6. 對比總結：REINFORCE vs Actor-Critic
    # ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  REINFORCE vs Actor-Critic 對比")
    print("=" * 60)
    rows = [
        ("網路數量",   "1（policy）",              "2（Actor + Critic）"),
        ("學習信號",   "normalize(G_t)",            "Advantage = G_t - V(s)"),
        ("Baseline",   "全局 mean（粗略）",          "V(s)（對應狀態的期望值）"),
        ("方差",       "高",                        "低（Critic 吸收基線方差）"),
        ("Critic 目標","無",                         "V(s) → G_t（MSE loss）"),
        ("更新時機",   "Episode 結束後",             "Episode 結束後（相同）"),
    ]
    print(f"  {'':14s} {'REINFORCE':30s} {'Actor-Critic':30s}")
    print("  " + "-" * 76)
    for label, pg, ac in rows:
        print(f"  {label:14s} {pg:30s} {ac:30s}")

    print("\n核心洞見：")
    print("  REINFORCE → 用 G_t 判斷：這段走得好不好（絕對值，方差大）")
    print("  A-C       → 用 A_t 判斷：這個動作比期望好多少（相對值，方差小）")
    print("  下一步：PPO 在 Actor-Critic 的基礎上加 clip，讓更新更保守穩定。")

    return agent


if __name__ == "__main__":
    agent = train()
    demo(lambda state: int(np.argmax(agent.actor.predict_probs(state))))
