"""
RL Hello World 2 — DQN on CartPole
====================================
從 GridWorld Q-table 進化到 Deep Q-Network

【為什麼 GridWorld 的做法在這裡行不通？】

  GridWorld：狀態是格子座標 (0~3, 0~3) → 只有 16 種狀態，Q-table 裝得下
  CartPole ：狀態是 4 個連續浮點數 → 無限多種狀態，Q-table 裝不下

  解法：用神經網路取代 Q-table
    輸入：4 個連續狀態值
    輸出：每個動作的 Q 值
    → 神經網路學會「泛化」，沒看過的狀態也能估出合理的 Q 值

【CartPole 環境】
  狀態 (4 個數值)：
    - 小車位置
    - 小車速度
    - 桿子角度
    - 桿子角速度
  動作：向左推 (0) 或向右推 (1)
  獎勵：每一步 +1（撐越久分越高）
  結束：桿子倒下 (>12度) 或小車出界

【DQN 對比 Q-table】

  Q-table：     Q(s, a) 直接查表
  DQN   ：     Q(s, a) = network(s)[a]   ← 用 NN 當函數近似器

  更新邏輯完全一樣，都是 Bellman：
    target = r + γ · max_a' Q(s', a')
    loss   = (target - Q(s, a))²    ← 用 MSE 當誤差，gradient descent 更新

【DQN 的兩個關鍵技巧】

  1. Experience Replay (經驗回放)
     把走過的 (s, a, r, s', done) 存進 buffer
     每次訓練從 buffer 隨機抽一批出來學
     → 打破資料的時間相關性，讓訓練更穩定

  2. Target Network (目標網路)
     準備兩個一樣的網路：main_net 和 target_net
     計算 target 時用 target_net（固定不動）
     每隔幾步才把 main_net 的參數複製給 target_net
     → 避免「追著自己的尾巴跑」導致訓練震盪
"""

import numpy as np
import random
from collections import deque
import gymnasium as gym

# ─────────────────────────────────────────────────────────
#  1. 簡單神經網路（純 numpy 實作，看清楚內部）
# ─────────────────────────────────────────────────────────

class SimpleNN:
    """
    兩層全連接神經網路
    架構：輸入(4) → 隱藏層(64, ReLU) → 輸出(2)

    為什麼要 ReLU？
      讓網路能學非線性函數。CartPole 的 Q 值跟狀態的關係是非線性的，
      純線性層無論疊幾層都只能學線性關係。
    """
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=2):
        # Xavier 初始化：讓各層輸出的方差接近 1，避免梯度消失/爆炸
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)

    def forward(self, x):
        """前向傳播：輸入狀態 → 輸出每個動作的 Q 值"""
        self.x   = x
        self.h   = np.maximum(0, x @ self.W1 + self.b1)   # ReLU
        self.out = self.h @ self.W2 + self.b2
        return self.out

    def backward(self, grad_out, lr):
        """反向傳播：用鏈式法則算梯度，更新權重"""
        # 輸出層梯度
        dW2 = self.h.T @ grad_out
        db2 = grad_out.sum(axis=0)

        # 隱藏層梯度（ReLU 的導數：正值為1，負值為0）
        grad_h = grad_out @ self.W2.T
        grad_h[self.h <= 0] = 0    # ReLU 反向

        dW1 = self.x.T @ grad_h
        db1 = grad_h.sum(axis=0)

        # Gradient descent 更新
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def copy_from(self, other):
        """複製另一個網路的參數（用於 target network 更新）"""
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()

    def predict(self, state):
        """單筆預測（inference 用）"""
        x = state.reshape(1, -1)
        return self.forward(x)[0]


# ─────────────────────────────────────────────────────────
#  2. Experience Replay Buffer
# ─────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    儲存過去的經驗 (s, a, r, s', done)
    訓練時隨機抽 batch，打破時間相關性

    為什麼要打破相關性？
      連續幾步的狀態很相似（t 時刻的狀態跟 t+1 幾乎一樣）
      如果連續拿來訓練，等於重複告訴網路同一件事，容易過擬合
      隨機抽樣讓每個 batch 的資料多樣化
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions),
                np.array(rewards), np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────────────────
#  3. DQN Agent
# ─────────────────────────────────────────────────────────

class DQNAgent:
    def __init__(self):
        self.main_net   = SimpleNN()    # 主網路：持續被訓練
        self.target_net = SimpleNN()    # 目標網路：定期同步，計算 target 用
        self.target_net.copy_from(self.main_net)

        self.replay_buffer = ReplayBuffer(capacity=10000)

        # 超參數
        self.gamma       = 0.99    # 折扣因子（CartPole 需要考慮長遠，設高一點）
        self.lr          = 0.001   # 學習率
        self.batch_size  = 64      # 每次訓練抽多少筆
        self.epsilon     = 1.0     # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update_freq = 10  # 每幾個 episode 同步一次 target net

    def choose_action(self, state):
        """ε-greedy：探索 or 利用"""
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        q_values = self.main_net.predict(state)
        return int(np.argmax(q_values))

    def train(self):
        """從 replay buffer 抽 batch，用 Bellman 更新網路"""
        if len(self.replay_buffer) < self.batch_size:
            return None   # buffer 還不夠，先跳過

        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)

        # 用 main_net 計算目前的 Q 值
        q_values = self.main_net.forward(states)          # shape: (batch, 2)

        # 用 target_net 計算下一步的最大 Q 值（Bellman target）
        next_q   = self.target_net.forward(next_states)   # shape: (batch, 2)
        max_next_q = np.max(next_q, axis=1)               # shape: (batch,)

        # target = r + γ · max Q(s', a')（終止狀態只有 r）
        targets = rewards + self.gamma * max_next_q * (1 - dones)

        # 只更新實際執行的那個動作的 Q 值，其他動作保持不變
        # （這樣 gradient 只流過被選中的動作）
        q_target = q_values.copy()
        q_target[np.arange(self.batch_size), actions] = targets

        # MSE loss 的梯度：2 * (預測 - 目標) / batch_size
        grad = 2 * (q_values - q_target) / self.batch_size

        # 反向傳播更新 main_net
        self.main_net.backward(grad, self.lr)

        # 計算 loss（純用來觀察訓練是否穩定）
        loss = np.mean((q_values - q_target) ** 2)
        return loss

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ─────────────────────────────────────────────────────────
#  4. 訓練迴圈
# ─────────────────────────────────────────────────────────

def train():
    env   = gym.make("CartPole-v1")
    agent = DQNAgent()

    EPISODES = 300
    scores   = []

    print("=" * 55)
    print("  RL Hello World 2 — DQN on CartPole")
    print("=" * 55)
    print("\n狀態空間：4 個連續數值（無法用 Q-table）")
    print("動作空間：向左(0) / 向右(1)")
    print("目標    ：撐超過 195 步 = 解決！")
    print(f"\n開始訓練 {EPISODES} 個 episodes...\n")

    for episode in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0
        losses = []

        while True:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 存入 replay buffer
            agent.replay_buffer.push(state, action, reward, next_state, float(done))

            # 訓練
            loss = agent.train()
            if loss is not None:
                losses.append(loss)

            total_reward += reward
            state = next_state

            if done:
                break

        agent.decay_epsilon()
        scores.append(total_reward)

        # 定期同步 target network
        if (episode + 1) % agent.target_update_freq == 0:
            agent.target_net.copy_from(agent.main_net)

        # 每 20 回合印進度
        if (episode + 1) % 20 == 0:
            avg_score = np.mean(scores[-20:])
            avg_loss  = np.mean(losses) if losses else 0
            solved    = "✓ 解決！" if avg_score >= 195 else ""
            print(f"Episode {episode+1:4d} | "
                  f"近20回合平均分: {avg_score:6.1f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"ε={agent.epsilon:.3f}  {solved}")

    env.close()

    # ─────────────────────────────────────────────────────
    #  5. 展示訓練完的 agent
    # ─────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  訓練完成！跑 5 次展示學到的策略")
    print("=" * 55)

    env = gym.make("CartPole-v1")
    for trial in range(5):
        state, _ = env.reset()
        steps = 0
        while True:
            q_values = agent.main_net.predict(state)
            action = int(np.argmax(q_values))
            state, _, terminated, truncated, _ = env.step(action)
            steps += 1
            if terminated or truncated:
                break
        result = "✓ 撐住了！" if steps >= 195 else f"倒了（{steps} 步）"
        print(f"  Trial {trial+1}：{steps} 步  {result}")

    env.close()

    # ─────────────────────────────────────────────────────
    #  6. 跟 GridWorld 的對比總結
    # ─────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  GridWorld Q-table vs CartPole DQN 對比")
    print("=" * 55)
    rows = [
        ("狀態空間",   "16 種（離散）",     "無限（連續浮點數）"),
        ("Q 函數",     "查表 Q[s][a]",       "神經網路 network(s)[a]"),
        ("更新方式",   "直接改 Q 值",        "gradient descent"),
        ("泛化能力",   "無（沒看過=不知道）","有（內插相似狀態）"),
        ("Bellman",    "一樣",               "一樣"),
        ("ε-greedy",  "一樣",               "一樣"),
    ]
    print(f"  {'':12s} {'GridWorld':22s} {'CartPole DQN':22s}")
    print("  " + "-" * 58)
    for label, gw, cp in rows:
        print(f"  {label:12s} {gw:22s} {cp:22s}")
    print("\n核心洞見：")
    print("  Bellman 更新邏輯完全沒變。")
    print("  唯一的差別是把 Q-table 換成了神經網路。")
    print("  這就是 Deep RL 的本質。")


if __name__ == "__main__":
    train()
