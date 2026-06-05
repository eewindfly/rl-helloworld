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

  ⚠️ 這也是整個系列「框架切換點」對齊「概念切換點」的地方：
     階段 1（GridWorld）狀態少 → Q-table → 純 numpy 就夠。
     階段 2 起狀態變連續 → 需要神經網路 → 改用 PyTorch。

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

────────────────────────────────────────────────────────────
【關於這版：從 numpy 手刻 → PyTorch】

  演算法（Bellman、replay、target net）一字未改，只有 NN 換成 PyTorch：

    numpy 版：自己寫 SimpleNN.forward / backward、手動算 MSE 梯度
              2*(pred-target)、手動只更新被選動作那一格。
    PyTorch ：nn.Module 定義網路，q.gather(1, actions) 取被選動作的 Q，
              F.mse_loss + loss.backward() + optimizer.step() 全自動。

  兩個 numpy 版要小心手刻、這裡 PyTorch 一行解決的地方：
    1. 「只更新被選動作的 Q」→ q_values.gather(1, actions)
    2. 「target 不要回傳梯度」→ with torch.no_grad(): 包住 target 計算
       （手刻版是靠「只把被選動作那格設成 target、其餘相減為 0」隱含達成）
"""

import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from utils import demo

# ─────────────────────────────────────────────────────────
#  1. Q 網路（PyTorch）
# ─────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """
    兩層全連接神經網路
    架構：輸入(4) → 隱藏層(64, ReLU) → 輸出(2 個動作的 Q 值)

    為什麼要 ReLU？
      讓網路能學非線性函數。CartPole 的 Q 值跟狀態的關係是非線性的，
      純線性層無論疊幾層都只能學線性關係。

    對比手刻 numpy 版：
      numpy 版自己寫 Xavier 初始化、forward 的 ReLU、backward 的鏈式法則。
      PyTorch 的 nn.Linear 已內建合理初始化，autograd 自動處理 backward。
    """
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        """前向傳播：狀態 → 每個動作的 Q 值"""
        return self.net(x)

    @torch.no_grad()
    def predict(self, state):
        """單筆推論：numpy 狀態 → numpy Q 值（給 choose_action / demo 用）"""
        x = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        return self.forward(x)[0].numpy()


# ─────────────────────────────────────────────────────────
#  2. Experience Replay Buffer
# ─────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    儲存過去的經驗 (s, a, r, s', terminated)
    訓練時隨機抽 batch，打破時間相關性

    ⚠️ 第 5 格存的是 terminated（桿子真的倒了），不是 done。
       truncated（撐到 500 步上限被截斷）不算終止，s' 仍有未來價值，
       target 要照常 bootstrap。和 actor_critic_td / ppo 的處理一致。

    為什麼要打破相關性？
      連續幾步的狀態很相似（t 時刻的狀態跟 t+1 幾乎一樣）
      如果連續拿來訓練，等於重複告訴網路同一件事，容易過擬合
      隨機抽樣讓每個 batch 的資料多樣化
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, terminated):
        self.buffer.append((state, action, reward, next_state, terminated))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, terminateds = zip(*batch)
        return (np.array(states), np.array(actions),
                np.array(rewards), np.array(next_states), np.array(terminateds))

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────────────────
#  3. DQN Agent
# ─────────────────────────────────────────────────────────

class DQNAgent:
    def __init__(self):
        self.main_net   = QNetwork()    # 主網路：持續被訓練
        self.target_net = QNetwork()    # 目標網路：定期同步，計算 target 用
        self.target_net.load_state_dict(self.main_net.state_dict())

        # Adam 取代手刻的 W -= lr * grad
        self.optimizer = torch.optim.Adam(self.main_net.parameters(), lr=0.001)

        self.replay_buffer = ReplayBuffer(capacity=10000)

        # 超參數
        self.gamma       = 0.99    # 折扣因子（CartPole 需要考慮長遠，設高一點）
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

        states, actions, rewards, next_states, terminateds = \
            self.replay_buffer.sample(self.batch_size)

        states      = torch.as_tensor(states,      dtype=torch.float32)
        actions     = torch.as_tensor(actions,     dtype=torch.long)
        rewards     = torch.as_tensor(rewards,     dtype=torch.float32)
        next_states = torch.as_tensor(next_states, dtype=torch.float32)
        terminateds = torch.as_tensor(terminateds, dtype=torch.float32)

        # 目前的 Q(s, a)：先算所有動作的 Q，再用 gather 取「實際執行的那個動作」
        #   gather 取代手刻版「q_target[arange, actions] = targets，其餘相減為 0」
        #   的技巧——只讓被選動作那一格參與 loss，梯度自然只流過它。
        q_values = self.main_net(states)                          # (batch, 2)
        q_sa     = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # (batch,)

        # Bellman target = r + γ · max Q(s', a')
        #   ⚠️ 只有 terminated（真終止）才把未來歸零，用 (1 - terminated)。
        #      truncated（撐到上限被截斷）桿子還立著，s' 仍有未來價值，
        #      照常 bootstrap——和 actor_critic_td / ppo 一致。
        #   用 no_grad 包住：target 是「固定目標」，不該對它回傳梯度。
        #   （手刻版是靠 target_net 不參與 backward 隱含做到這件事。）
        with torch.no_grad():
            next_q     = self.target_net(next_states)            # (batch, 2)
            max_next_q = next_q.max(dim=1).values                # (batch,)
            targets    = rewards + self.gamma * max_next_q * (1 - terminateds)

        # MSE loss + 自動反向傳播 + 自動更新
        loss = F.mse_loss(q_sa, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def sync_target(self):
        """把 main_net 的參數複製給 target_net"""
        self.target_net.load_state_dict(self.main_net.state_dict())

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
    print("  RL Hello World 2 — DQN on CartPole [PyTorch]")
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
            done = terminated or truncated   # 只用來結束這個 episode 的迴圈

            # 存入 replay buffer：第 5 格存 terminated（不是 done）。
            # 截斷 truncated 不算終止，target 要照常 bootstrap（見 train()）。
            agent.replay_buffer.push(state, action, reward, next_state, float(terminated))

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
            agent.sync_target()

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

    return agent


if __name__ == "__main__":
    agent = train()
    demo(lambda state: int(np.argmax(agent.main_net.predict(state))))
