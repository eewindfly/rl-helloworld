"""
RL Hello World 3 — Policy Gradient (REINFORCE) on CartPole
============================================================
從 DQN（學 Q 值）進化到 Policy Gradient（直接學動作機率）

【DQN 的做法回顧】

  DQN 學的是 Q(s, a)：每個動作的「預期總獎勵」
  選動作：argmax Q(s, a)  ← 間接，先估值再推動作

【Policy Gradient 的做法】

  直接學 policy π(a|s)：在狀態 s 下，每個動作的「機率」
  選動作：依照機率分佈取樣  ← 直接輸出機率

  網路輸出 softmax → 機率分佈
    [0.3, 0.7] → 30% 向左，70% 向右

【REINFORCE 演算法】

  核心思路：
    做出好結果的動作 → 提高它的機率
    做出差結果的動作 → 降低它的機率

  怎麼定義「好不好」？
    G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...
    = 從第 t 步開始到結尾的累積折扣獎勵（Monte Carlo return）

  更新公式：
    ∇J = Σ_t  ∇log π(a_t | s_t) × G_t

    直覺解釋：
      G_t 大（這一段走得好）→ 增大 log π（提高這些動作的機率）
      G_t 小（走得差）       → 減小 log π（降低這些動作的機率）

【REINFORCE vs DQN 的關鍵差異】

  | 項目           | DQN                    | REINFORCE              |
  |----------------|------------------------|------------------------|
  | 學的是         | Q(s,a) 數值            | π(a|s) 機率            |
  | 選動作         | argmax（確定性）        | 取樣（隨機性）          |
  | 訓練時機       | 每一步（off-policy）   | 整個 episode 結束後     |
  | 資料使用       | replay buffer 重複用   | 跑完就丟（on-policy）   |
  | 方差           | 低                     | 高（Monte Carlo）       |

【為什麼要學 Policy Gradient？】

  Actor-Critic 和 PPO 全部建立在這個基礎上。
  RLHF（ChatGPT 的訓練方式）用的是 PPO，而 PPO 是 Policy Gradient 的延伸。
"""

import numpy as np
import gymnasium as gym


# ─────────────────────────────────────────────────────────
#  1. Policy Network（純 numpy）
# ─────────────────────────────────────────────────────────

class PolicyNetwork:
    """
    Policy 網路：輸入狀態，輸出動作機率
    架構：輸入(4) → 隱藏層(64, ReLU) → 輸出(2, Softmax)

    和 DQN 的網路差異：
      DQN：輸出 Q 值（任意實數）
      PG ：輸出機率（0~1，加總為 1）→ 用 softmax 保證
    """
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=2):
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)

    def softmax(self, x):
        """數值穩定的 softmax：先減最大值防止 overflow"""
        x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / exp_x.sum(axis=-1, keepdims=True)

    def forward(self, x):
        """前向傳播，回傳動作機率"""
        self.x  = x
        self.h  = np.maximum(0, x @ self.W1 + self.b1)   # ReLU
        logits  = self.h @ self.W2 + self.b2
        self.probs = self.softmax(logits)
        return self.probs

    def backward(self, grad_logits, lr):
        """
        反向傳播
        grad_logits：loss 對 softmax 輸入（logits）的梯度
        """
        # 輸出層
        dW2 = self.h.T @ grad_logits
        db2 = grad_logits.sum(axis=0)

        # 隱藏層（ReLU 反向）
        grad_h = grad_logits @ self.W2.T
        grad_h[self.h <= 0] = 0

        dW1 = self.x.T @ grad_h
        db1 = grad_h.sum(axis=0)

        # Gradient ascent（我們在最大化目標，所以加而不是減）
        # 這裡統一用 -= 並讓外部傳入負梯度（等效）
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def predict_probs(self, state):
        """單筆推論：輸入狀態，輸出動作機率"""
        x = state.reshape(1, -1)
        return self.forward(x)[0]


# ─────────────────────────────────────────────────────────
#  2. Policy Gradient Agent (REINFORCE)
# ─────────────────────────────────────────────────────────

class PGAgent:
    def __init__(self):
        self.policy_net  = PolicyNetwork()
        self.best_net    = PolicyNetwork()   # 保存訓練中最佳的 policy
        self.best_score  = -float('inf')

        # 超參數
        self.gamma = 0.99   # 折扣因子
        self.lr    = 0.01   # 學習率

        # 存放多條軌跡（每條是一個 episode）
        self.trajectories = []      # list of (states, actions, rewards)
        self._cur_states  = []      # 當前 episode 暫存
        self._cur_actions = []
        self._cur_rewards = []

    def update_best(self, avg_score):
        """如果近期平均分創新高，保存當前 policy"""
        if avg_score > self.best_score:
            self.best_score = avg_score
            self.best_net.W1 = self.policy_net.W1.copy()
            self.best_net.b1 = self.policy_net.b1.copy()
            self.best_net.W2 = self.policy_net.W2.copy()
            self.best_net.b2 = self.policy_net.b2.copy()

    def choose_action(self, state):
        """
        根據 policy 輸出的機率分佈取樣動作

        為什麼用取樣而不是 argmax？
          取樣保留了「探索性」：機率低的動作偶爾也會被選到
          argmax 是確定性的，會困在局部最優

        注意：測試時可以改用 argmax（exploitation only）
        """
        probs = self.policy_net.predict_probs(state)
        action = np.random.choice(len(probs), p=probs)
        return action

    def store(self, state, action, reward):
        """記錄這一步的資料"""
        self._cur_states.append(state)
        self._cur_actions.append(action)
        self._cur_rewards.append(reward)

    def end_episode(self):
        """Episode 結束，把這條軌跡存起來"""
        self.trajectories.append((
            list(self._cur_states),
            list(self._cur_actions),
            list(self._cur_rewards),
        ))
        self._cur_states  = []
        self._cur_actions = []
        self._cur_rewards = []

    def compute_returns(self, rewards):
        """
        計算單條軌跡每一步的 Monte Carlo Return G_t

        G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... + γ^{T-t}·r_T
        """
        T = len(rewards)
        returns = np.zeros(T)
        G = 0.0
        for t in reversed(range(T)):
            G = rewards[t] + self.gamma * G
            returns[t] = G
        return returns

    def update(self):
        """
        用收集到的 N 條軌跡更新 policy

        對應公式：∇R̄_θ ≈ (1/N) Σ_n Σ_t R(τ^n) ∇log π(a_t^n | s_t^n)

        每條軌跡先各自算 return，N 條全部收集完後統一 normalize。
        除以 N 對應公式裡的 1/N（估計期望值）。

        REINFORCE 梯度公式：
          ∇J = Σ_t  ∇ log π(a_t | s_t) × G_t

        ∇ log π(a_t | s_t) 對 softmax logits 的梯度：
          令 p = softmax(logits)
          ∂ log p[a] / ∂ logits = (1_{i=a} - p[i])
          也就是：one-hot(a) - p

        Loss（我們最大化 J，等效於最小化 -J）：
          對 logits 的梯度 = -(one_hot(a) - p) × G_t
        """
        N = len(self.trajectories)

        # 每條軌跡各自算 return，再合併
        all_states   = []
        all_actions  = []
        all_returns  = []

        for states, actions, rewards in self.trajectories:
            returns = self.compute_returns(rewards)
            all_states.extend(states)
            all_actions.extend(actions)
            all_returns.extend(returns)

        states  = np.array(all_states)
        actions = np.array(all_actions)
        returns = np.array(all_returns)

        # N 條軌跡合併後統一 normalize，保留跨 episode 的相對差異
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        T = len(states)

        # 前向傳播
        probs = self.policy_net.forward(states)   # (T, 2)

        # 計算梯度：∂(-J) / ∂logits_t = -(one_hot(a_t) - probs_t) × G_t
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(T), actions] = 1.0

        grad_logits = -(one_hot - probs) * returns.reshape(-1, 1)
        grad_logits /= N    # 對應公式的 1/N

        # 反向傳播更新 policy
        self.policy_net.backward(grad_logits, self.lr)

        # 清空（on-policy：用完就丟）
        self.trajectories = []


# ─────────────────────────────────────────────────────────
#  3. 訓練迴圈
# ─────────────────────────────────────────────────────────

def train():
    env   = gym.make("CartPole-v1")
    agent = PGAgent()

    EPISODES  = 1000
    N_UPDATES = 4     # 每收集幾條軌跡才更新一次（N in the formula）
    scores    = []

    print("=" * 60)
    print("  RL Hello World 3 — Policy Gradient (REINFORCE)")
    print("=" * 60)
    print("\n核心概念：直接學動作機率，不再學 Q 值")
    print(f"更新時機：每收集 {N_UPDATES} 條軌跡後更新（Monte Carlo，N={N_UPDATES}）")
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

        agent.end_episode()
        scores.append(total_reward)

        # 每收集 N_UPDATES 條軌跡才更新一次
        if (episode + 1) % N_UPDATES == 0:
            agent.update()

        # 每 50 回合印進度，並更新最佳 policy
        if (episode + 1) % 50 == 0:
            avg_score = np.mean(scores[-50:])
            agent.update_best(avg_score)
            solved    = "✓ 解決！" if avg_score >= 195 else ""
            print(f"Episode {episode+1:4d} | "
                  f"近50回合平均分: {avg_score:6.1f}  {solved}")

    env.close()

    # ─────────────────────────────────────────────────────
    #  4. 展示訓練完的 agent
    # ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  訓練完成！用最佳 policy（平均分 {agent.best_score:.1f}）跑 5 次展示")
    print("=" * 60)

    env = gym.make("CartPole-v1")
    for trial in range(5):
        state, _ = env.reset()
        steps = 0
        while True:
            # 展示時用最佳 policy + argmax（確定性）
            probs  = agent.best_net.predict_probs(state)
            action = int(np.argmax(probs))
            state, _, terminated, truncated, _ = env.step(action)
            steps += 1
            if terminated or truncated:
                break
        result = "✓ 撐住了！" if steps >= 195 else f"倒了（{steps} 步）"
        print(f"  Trial {trial+1}：{steps} 步  {result}")

    env.close()

    # ─────────────────────────────────────────────────────
    #  5. 對比總結：DQN vs Policy Gradient
    # ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DQN vs Policy Gradient 對比")
    print("=" * 60)
    rows = [
        ("學的目標",   "Q(s,a) 數值",          "π(a|s) 機率"),
        ("輸出層",     "線性（任意實數）",       "Softmax（機率）"),
        ("選動作",     "argmax Q（確定性）",     "取樣（隨機性）"),
        ("訓練時機",   "每一步",                "Episode 結束後"),
        ("資料重用",   "Replay buffer（可重用）","On-policy（跑完就丟）"),
        ("Return",     "Bellman（一步 TD）",     "Monte Carlo（整段）"),
        ("方差",       "低",                    "高（需 normalize）"),
    ]
    print(f"  {'':14s} {'DQN':24s} {'Policy Gradient':24s}")
    print("  " + "-" * 64)
    for label, dqn, pg in rows:
        print(f"  {label:14s} {dqn:24s} {pg:24s}")

    print("\n核心洞見：")
    print("  DQN  → 學 Q 值 → argmax → 動作（間接）")
    print("  PG   → 直接學機率 → 取樣 → 動作（直接）")
    print("  下一步：Actor-Critic 把兩者結合，用 Critic 降低 PG 的高方差。")

    return agent


def demo(agent):
    """開視窗展示最佳 policy 的實際表現"""
    print("\n開啟視覺化視窗，按 Ctrl+C 結束...")
    env = gym.make("CartPole-v1", render_mode="human")
    try:
        while True:
            state, _ = env.reset()
            steps = 0
            while True:
                probs  = agent.best_net.predict_probs(state)
                action = int(np.argmax(probs))
                state, _, terminated, truncated, _ = env.step(action)
                steps += 1
                if terminated or truncated:
                    print(f"  {steps} 步")
                    break
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    agent = train()
    demo(agent)
