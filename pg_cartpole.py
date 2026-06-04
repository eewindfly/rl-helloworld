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

────────────────────────────────────────────────────────────
【關於這版：從 numpy 手刻 → PyTorch】

  階段 2~5 改用 PyTorch。和先前純 numpy 版本相比，演算法一個字都沒變，
  只有「梯度怎麼算」這件事換了實作：

    numpy 版：自己推 softmax 的梯度 (one_hot - probs)、自己寫 backward()、
              手動 W -= lr * grad。
    PyTorch ：forward 只寫到 loss，剩下 loss.backward() 由 autograd 自動
              算出所有梯度，optimizer.step() 自動更新。

  REINFORCE 在 PyTorch 裡乾淨到只剩三行：
      log_probs = Categorical(logits).log_prob(actions)
      loss      = -(log_probs * returns).mean()   ← 最大化 J = 最小化 -J
      loss.backward(); opt.step()

  你手刻過一次、知道 (one_hot - probs) 是怎麼來的，現在可以放心把它交給
  autograd —— 這就是換框架的全部意義。
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym
from utils import demo


# ─────────────────────────────────────────────────────────
#  1. Policy Network（PyTorch）
# ─────────────────────────────────────────────────────────

class PolicyNetwork(nn.Module):
    """
    Policy 網路：輸入狀態，輸出動作的 logits
    架構：輸入(4) → 隱藏層(64, ReLU) → 輸出(2)

    和 DQN 的網路差異：
      DQN：輸出 Q 值（任意實數，直接拿來比大小）
      PG ：輸出 logits → 經 softmax 變成機率（0~1，加總為 1）

    為什麼 forward 回傳 logits 而不是 probs？
      PyTorch 的 Categorical(logits=...) 接 logits，內部會用數值穩定的
      log-softmax，比我們先 softmax 再取 log 更安全（不會 log(0)）。
      手刻 numpy 版自己做了「減最大值」防 overflow，這裡 PyTorch 幫你做掉了。
    """
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        """前向傳播：狀態 → 每個動作的 logits（未經 softmax）"""
        return self.net(x)

    @torch.no_grad()
    def predict_probs(self, state):
        """單筆推論：numpy 狀態 → numpy 動作機率（給 choose_action / demo 用）"""
        x = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)[0].numpy()


# ─────────────────────────────────────────────────────────
#  2. Policy Gradient Agent (REINFORCE)
# ─────────────────────────────────────────────────────────

class PGAgent:
    def __init__(self):
        self.policy_net = PolicyNetwork()

        # 超參數
        self.gamma = 0.99   # 折扣因子
        self.lr    = 0.01   # 學習率

        # PyTorch 用 optimizer 取代手刻的 W -= lr * grad。
        # Adam 是 RL 的事實標準（Spinning Up / CleanRL 全用它）：
        # 它對每個參數自適應調整步長，比純 SGD 穩、好調。
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)

        # 存放多條軌跡（每條是一個 episode）
        self.trajectories = []      # list of (states, actions, rewards)
        self._cur_states  = []      # 當前 episode 暫存
        self._cur_actions = []
        self._cur_rewards = []

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
        計算單條軌跡每一步的 Monte Carlo return（reward-to-go）G_t

        G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... + γ^{T-t}·r_T

        「reward-to-go」= 從第 t 步開始還能拿到多少獎勵。
        用 G_t 而非整條軌跡的總回報 R(τ)，是因為 t 步之前的獎勵
        跟第 t 步選什麼動作無關，納入只會增加梯度估計的方差，
        拿掉後期望值不變但方差更小。
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

        REINFORCE 梯度公式：
          ∇J = Σ_t  ∇ log π(a_t | s_t) × G_t

        【手刻 numpy → PyTorch 的對照】

          numpy 版要自己做這些：
            probs = softmax(logits)
            grad_logits = -(one_hot(a) - probs) × G_t      ← 自己推的 softmax 梯度
            policy_net.backward(grad_logits, lr)           ← 自己寫的反向傳播

          PyTorch 版只要寫到「loss 長什麼樣」，梯度自動算：
            log_probs = Categorical(logits).log_prob(actions)
            loss      = -(log_probs × G_t).sum() / N       ← 對應公式的 1/N
            loss.backward()                                 ← autograd 自動算 ∇
            optimizer.step()                                ← 自動更新

          (one_hot - probs) 沒有消失，它就是 autograd 對 log_prob 求導的結果，
          只是現在不用你手算了。
        """
        # 每條軌跡各自算 return，再合併
        N = len(self.trajectories)   # 軌跡數，對應公式裡的 N
        all_states   = []
        all_actions  = []
        all_returns  = []

        for states, actions, rewards in self.trajectories:
            returns = self.compute_returns(rewards)
            all_states.extend(states)
            all_actions.extend(actions)
            all_returns.extend(returns)

        states  = torch.as_tensor(np.array(all_states),  dtype=torch.float32)
        actions = torch.as_tensor(np.array(all_actions), dtype=torch.long)
        returns = torch.as_tensor(np.array(all_returns), dtype=torch.float32)

        # N 條軌跡合併後統一 normalize，保留跨 episode 的相對差異
        # 注意：用全局 mean 當 baseline 有已知的系統性問題——
        #   G_t 天然隨時間遞減（後面步數少，累積獎勵小），
        #   導致後期動作相對 mean 偏低，被系統性懲罰，
        #   即使整條軌跡走得好也一樣。
        # 正確做法是用 V(s) 當 baseline（Advantage = G_t - V(s)），
        # 針對每個狀態估出合理期望，這就是 Actor-Critic 要解決的問題。
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # 前向 → 機率分佈 → 取出實際動作的 log π(a|s)
        logits    = self.policy_net(states)             # (T, 2)
        dist      = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)              # (T,)

        # Loss = -(1/N) Σ_n Σ_t log π(a_t|s_t) × G_t，對應公式的 1/N。
        # .mean() 會除以總 timestep 數 T（= N × avg_episode_len），
        # 和公式的 /N 相差一個 avg_episode_len，所以用 .sum() / N。
        loss = -(log_probs * returns).sum() / N

        # 一行反向傳播 + 一行更新，取代整個手刻 backward()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 清空（on-policy：用完就丟）
        self.trajectories = []


# ─────────────────────────────────────────────────────────
#  3. 訓練迴圈
# ─────────────────────────────────────────────────────────

def train():
    env   = gym.make("CartPole-v1")
    agent = PGAgent()

    BATCH_EPISODES = 4     # 每收集幾條軌跡才更新一次（公式裡的 N，也是 batch size）
    NUM_UPDATES    = 250   # 總共要更新幾次（真正決定學不學得起來的量）
    EPISODES       = BATCH_EPISODES * NUM_UPDATES
    scores         = []

    print("=" * 60)
    print("  RL Hello World 3 — Policy Gradient (REINFORCE) [PyTorch]")
    print("=" * 60)
    print("\n核心概念：直接學動作機率，不再學 Q 值")
    print(f"更新時機：每收集 {BATCH_EPISODES} 條軌跡後更新（Monte Carlo，N={BATCH_EPISODES}），共 {NUM_UPDATES} 次")
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

        # 每收集 BATCH_EPISODES 條軌跡才更新一次
        if (episode + 1) % BATCH_EPISODES == 0:
            agent.update()

        # 每 50 回合印進度
        if (episode + 1) % 50 == 0:
            avg_score = np.mean(scores[-50:])
            solved    = "✓ 解決！" if avg_score >= 195 else ""
            print(f"Episode {episode+1:4d} | "
                  f"近50回合平均分: {avg_score:6.1f}  {solved}")

    env.close()

    # ─────────────────────────────────────────────────────
    #  4. 展示訓練完的 agent
    # ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  訓練完成！跑 5 次展示")
    print("=" * 60)

    env = gym.make("CartPole-v1")
    for trial in range(5):
        state, _ = env.reset()
        steps = 0
        while True:
            probs  = agent.policy_net.predict_probs(state)
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


if __name__ == "__main__":
    agent = train()
    demo(lambda state: int(np.argmax(agent.policy_net.predict_probs(state))))
