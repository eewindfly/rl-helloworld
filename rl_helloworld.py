"""
RL Hello World — Q-Learning on GridWorld
=========================================
用最簡單的方式理解強化學習 (Reinforcement Learning) 的核心概念

核心概念對照表：
  State  (狀態)  → Agent 目前在格子地圖的哪個位置
  Action (動作)  → 上、下、左、右 四個方向
  Reward (獎勵)  → 到達終點 +10，踩到障礙 -5，每走一步 -0.1
  Policy (策略)  → 在每個格子「應該往哪走」的決策規則
  Q(s,a) (Q值)   → 預估「在狀態 s 採取動作 a，未來能拿到多少獎勵」

地圖 (4x4)：
  S . . .
  . X . .
  . . X .
  . . . G

  S = 起點 (0,0)
  G = 終點 (3,3)，到達得 +10 獎勵
  X = 障礙物，踩到得 -5 懲罰並結束這回合
  . = 空地，每走一步 -0.1 (鼓勵快點到達)
"""

import numpy as np
import random

# ─────────────────────────────────────────
#  1. 定義環境 (GridWorld)
# ─────────────────────────────────────────

GRID_SIZE = 4

# 地圖元素
START    = (0, 0)
GOAL     = (3, 3)
OBSTACLES = {(1, 1), (2, 2)}

# 動作對應的移動方向 (row_delta, col_delta)
ACTIONS = {
    0: (-1,  0),  # 上
    1: ( 1,  0),  # 下
    2: ( 0, -1),  # 左
    3: ( 0,  1),  # 右
}
ACTION_NAMES = {0: "↑", 1: "↓", 2: "←", 3: "→"}


def step(state, action):
    """
    環境的核心函式：給定目前狀態和動作，回傳 (next_state, reward, done)

    這就是 RL 的「環境」部分：
      - Agent 做動作 → 環境回應新狀態 + 獎勵
    """
    row, col = state
    dr, dc = ACTIONS[action]
    new_row = row + dr
    new_col = col + dc

    # 撞牆 → 原地不動
    if not (0 <= new_row < GRID_SIZE and 0 <= new_col < GRID_SIZE):
        new_row, new_col = row, col

    next_state = (new_row, new_col)

    # 計算獎勵
    if next_state == GOAL:
        return next_state, +10.0, True   # 到達終點！
    elif next_state in OBSTACLES:
        return next_state, -5.0, True    # 踩到障礙，回合結束
    else:
        return next_state, -0.1, False   # 普通移動，小懲罰促使快點到達


# ─────────────────────────────────────────
#  2. Q-Learning Agent
# ─────────────────────────────────────────
"""
Q-Learning 的核心想法：
  維護一張 Q-table：每個 (狀態, 動作) 組合 → 預期獎勵

  每走一步就更新 Q 值：
    Q(s,a) ← Q(s,a) + α × [r + γ × max Q(s',a') - Q(s,a)]
               ↑學習率      ↑即時獎勵  ↑折扣因子×未來最大獎勵

  白話：「這次實際得到的」比「原本預期的」好還是差？
        → 依據差距調整 Q 值，慢慢學到最好的策略
"""

# 超參數 (Hyperparameters)
ALPHA   = 0.1    # 學習率 (Learning Rate)：每次更新幅度，太大容易震盪、太小學很慢
GAMMA   = 0.9    # 折扣因子 (Discount Factor)：0=只看即時獎勵, 1=完全考慮未來
EPSILON = 1.0    # 探索率 (Exploration Rate)：1=全部隨機探索, 0=完全按策略行動
EPSILON_DECAY = 0.995  # 每回合 epsilon 衰減，讓 Agent 從「多探索」轉到「多利用」
EPSILON_MIN  = 0.01
EPISODES = 1000  # 訓練回合數
MAX_STEPS = 50   # 每回合最多走幾步（防止無限迴圈）


# Q-table：shape = [4][4][4 actions]，初始化為 0
# q_table[row][col][action] = 預期 Q 值
q_table = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))


def choose_action(state, epsilon):
    """
    Epsilon-Greedy 策略：
      以 epsilon 的機率「隨機探索」(Exploration)
      以 1-epsilon 的機率「選最佳動作」(Exploitation)

    為什麼需要探索？
      如果一開始就貪心選最好的，可能錯過更好的路徑
      就像你去新城市，要先亂逛才知道哪家餐廳最好
    """
    if random.random() < epsilon:
        return random.randint(0, len(ACTIONS) - 1)  # 探索：隨機
    else:
        row, col = state
        return int(np.argmax(q_table[row][col]))     # 利用：選 Q 值最高的


def update_q(state, action, reward, next_state, done):
    """
    Q-Learning 更新公式：
      Q(s,a) ← Q(s,a) + α × [target - Q(s,a)]

    target = r + γ × max_a' Q(s', a')   (如果 done 則 target = r)
    """
    row, col = state
    nr, nc = next_state

    current_q = q_table[row][col][action]

    if done:
        target = reward  # 終止狀態沒有「未來」
    else:
        target = reward + GAMMA * np.max(q_table[nr][nc])

    # 更新 Q 值
    q_table[row][col][action] = current_q + ALPHA * (target - current_q)


# ─────────────────────────────────────────
#  3. 訓練迴圈
# ─────────────────────────────────────────

print("=" * 50)
print("  RL Hello World — Q-Learning on GridWorld")
print("=" * 50)
print(f"\n地圖大小: {GRID_SIZE}×{GRID_SIZE}")
print(f"起點: {START}  終點: {GOAL}  障礙: {OBSTACLES}")
print(f"\n開始訓練 {EPISODES} 個 episodes...\n")

epsilon = EPSILON
total_rewards = []
success_count = 0

for episode in range(EPISODES):
    state = START   # 每回合從起點開始
    total_reward = 0

    for step_num in range(MAX_STEPS):
        # 1) Agent 選擇動作
        action = choose_action(state, epsilon)

        # 2) 環境執行動作，回傳結果
        next_state, reward, done = step(state, action)

        # 3) Agent 學習：更新 Q-table
        update_q(state, action, reward, next_state, done)

        # 累積獎勵、移動到下一狀態
        total_reward += reward
        state = next_state

        if done:
            if next_state == GOAL:
                success_count += 1
            break

    total_rewards.append(total_reward)

    # 衰減探索率
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    # 每 100 回合印一次進度
    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(total_rewards[-100:])
        print(f"Episode {episode+1:4d} | "
              f"近100回合平均獎勵: {avg_reward:6.2f} | "
              f"成功率: {success_count/100*100:.0f}% | "
              f"ε={epsilon:.3f}")
        success_count = 0


# ─────────────────────────────────────────
#  4. 展示學到的最佳路徑
# ─────────────────────────────────────────

def show_learned_policy():
    """把 Q-table 學到的最佳策略畫在地圖上"""
    print("\n" + "=" * 50)
    print("  學到的最佳策略 (每格顯示最佳動作)")
    print("=" * 50)
    for row in range(GRID_SIZE):
        line = ""
        for col in range(GRID_SIZE):
            pos = (row, col)
            if pos == GOAL:
                line += "  G  "
            elif pos in OBSTACLES:
                line += "  X  "
            elif pos == START:
                line += "  S  "
            else:
                best_action = int(np.argmax(q_table[row][col]))
                line += f"  {ACTION_NAMES[best_action]}  "
        print(line)
    print()


def show_best_path():
    """從起點跑一次貪心路徑（純利用，ε=0）"""
    print("=" * 50)
    print("  最佳路徑模擬 (ε=0，純貪心)")
    print("=" * 50)

    state = START
    path = [state]
    total_reward = 0

    for _ in range(MAX_STEPS):
        row, col = state
        action = int(np.argmax(q_table[row][col]))
        next_state, reward, done = step(state, action)
        path.append(next_state)
        total_reward += reward
        state = next_state
        if done:
            break

    # 畫地圖 + 路徑
    path_set = set(path)
    for row in range(GRID_SIZE):
        line = ""
        for col in range(GRID_SIZE):
            pos = (row, col)
            if pos == GOAL:
                line += " G "
            elif pos in OBSTACLES:
                line += " X "
            elif pos == START:
                line += " S "
            elif pos in path_set:
                line += " · "
            else:
                line += " . "
        print(line)

    print(f"\n路徑: {' → '.join(str(p) for p in path)}")
    print(f"步數: {len(path)-1}  總獎勵: {total_reward:.1f}")
    result = "✓ 成功到達終點！" if path[-1] == GOAL else "✗ 未到達終點"
    print(f"結果: {result}")


show_learned_policy()
show_best_path()

print("\n" + "=" * 50)
print("  Q-Table 部分數值 (前兩列)")
print("=" * 50)
print("格式：[上, 下, 左, 右]")
for row in range(2):
    for col in range(GRID_SIZE):
        vals = q_table[row][col]
        print(f"  ({row},{col}): [{vals[0]:5.2f}, {vals[1]:5.2f}, {vals[2]:5.2f}, {vals[3]:5.2f}]")
