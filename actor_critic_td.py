"""
RL Hello World 4b — Actor-Critic (A2C) TD 版 on CartPole
=========================================================
和 actor_critic.py 的唯一差異：把 MC return 換成 TD target。

【MC 版 vs TD 版的核心差異】

  MC 版（actor_critic.py）：
    - G_t = 從 t 步到 episode 結尾的累積折扣獎勵
    - 需要等整個 episode 跑完才能算
    - 無 bias，但方差高（後面很多隨機步驟的影響都算進來）

  TD 版（本檔）：
    - TD target = r_t + γ × V(s_{t+1})
    - 每步立刻就能算，不需要等 episode 結束
    - 有 bias（V 不準就偏），但方差低（只往前看一步）
    - terminal step：V(s_{t+1}) = 0（沒有下一個狀態）

【TD error = Advantage】

  δ_t = r_t + γ × V(s_{t+1}) - V(s_t)
      = TD target - V(s_t)

  這就是 TD 版的 Advantage。
  直接用 δ_t 更新 Actor：方差小，但因 V 不準而有 bias。

  MC Advantage：A_t = G_t - V(s_t)         ← 無 bias，方差高
  TD Advantage：δ_t = r_t + γV(s') - V(s)  ← 有 bias，方差低

【程式碼改動（相比 actor_critic.py）】

  1. store() 多存 next_state 和 done
  2. 拿掉 compute_returns()（不再需要 G_t）
  3. update() 裡：
       TD target  = r + γ × V(s')，terminal 時 V(s') = 0
       Advantage  = TD target - V(s)
       Critic loss = MSE(V(s), TD target)
  4. 更新時機不變：仍是 episode 結束後 batch 更新
     （也可改成每步更新，但 batch 較穩定、方便對比）
"""

import numpy as np
import gymnasium as gym
from utils import demo
from pg_cartpole import PolicyNetwork as ActorNetwork
from actor_critic import CriticNetwork   # Critic 架構完全相同，直接複用


# ─────────────────────────────────────────────────────────
#  Actor-Critic TD Agent
# ─────────────────────────────────────────────────────────

class ACTDAgent:
    def __init__(self):
        self.actor  = ActorNetwork()
        self.critic = CriticNetwork()

        self.gamma     = 0.99
        # ⚠️ 和 MC 版唯一要動的東西：學習率必須重調，不能照抄 MC 版！
        #   公式（advantage、critic target）完全照教科書，沒有任何額外技巧。
        #   原因見下方 update() 的「為何 MC 版的超參數搬過來會學不起來」。
        #   - actor_lr 調大（0.0005 → 0.02，約 40x）：
        #       TD advantage = δ 量級只有 O(1)，MC advantage = G_t - V 是 O(數十)，
        #       照抄 0.0005 等於 actor 幾乎不動。
        #   - critic_lr 調大（0.001 → 0.05）：
        #       TD 是 bootstrap，critic 要夠準 advantage 才有意義，
        #       必須讓 V(s) 快點學起來。
        self.actor_lr  = 0.02
        self.critic_lr = 0.05

        # 比 MC 版多存 next_state 和 done
        self._states      = []
        self._actions     = []
        self._rewards     = []
        self._next_states = []
        self._dones       = []

    def choose_action(self, state):
        probs = self.actor.predict_probs(state)
        return np.random.choice(len(probs), p=probs)

    def store(self, state, action, reward, next_state, done):
        self._states.append(state)
        self._actions.append(action)
        self._rewards.append(reward)
        self._next_states.append(next_state)
        self._dones.append(done)

    def update(self):
        states      = np.array(self._states)       # (T, 4)
        actions     = np.array(self._actions)      # (T,)
        rewards     = np.array(self._rewards)      # (T,)
        next_states = np.array(self._next_states)  # (T, 4)
        dones       = np.array(self._dones)        # (T,) bool

        T = len(rewards)
        N = int(dones.sum())   # 完整軌跡數（和 MC 版的 N 語意相同）

        # ── 步驟 1：算 TD target ──────────────────────────────
        #
        #   TD target = r + γ × V(s')
        #   terminal step：done=True，V(s') = 0（沒有下一狀態）
        #
        next_values = self.critic.forward(next_states)   # (T,)
        next_values[dones] = 0.0                         # terminal mask
        td_targets = rewards + self.gamma * next_values  # (T,)

        # ── 步驟 2：Critic 估 V(s) ───────────────────────────
        values = self.critic.forward(states)             # (T,)

        # ── 步驟 3：TD Advantage（= TD error δ）──────────────
        #
        #   δ_t = r_t + γ V(s') - V(s)
        #       = TD target - V(s)
        #
        #   對比 MC：Advantage = G_t - V(s)
        #   只是把 G_t 換成 TD target，其餘完全相同。
        #
        advantage = td_targets - values   # (T,)

        # ── 步驟 4：更新 Critic ──────────────────────────────
        #   讓 V(s) → TD target（而非 MC 版的 G_t）
        #   Loss = MSE(V(s), TD target)
        grad_v = (values - td_targets) / T
        self.critic.backward(grad_v, self.critic_lr)

        # ── 步驟 5：更新 Actor ───────────────────────────────
        #   和 MC 版完全相同，只是 advantage 換成 TD error δ。
        #   公式純教科書，沒有 normalize、沒有任何額外技巧。
        #
        #   ⚠️ 為何 MC 版的超參數搬過來會「學不起來」？（已在上方調 lr 修正）
        #
        #   CartPole 每一步（含倒下那步）reward 都是 +1。初期 critic 還沒學會，
        #   V(s) ≈ 0，於是 advantage = r + γV(s') - V(s) ≈ +1 對「每一步」都成立，
        #   連倒下那一步的 advantage 都是正的 → actor 被告知「每個動作都很好」，
        #   完全學不到東西（原 lr 實測 avg 卡在 ~13）。
        #
        #   MC 版為何沒這問題？因為 MC advantage = G_t - V(s)，
        #   主導項是真實 return G_t（O(數十)），就算 critic 很爛、符號也對得到。
        #   TD advantage 完全由 critic 兩次輸出相減而來，critic 不準 → advantage 是雜訊；
        #   且量級只有 O(1)（≈單一步的 δ），所以 actor_lr 必須調大、critic_lr 也要夠快。
        #
        #   代價：純 δ 不做 normalize，訓練會比 MC 版抖（偶爾回檔），
        #         這正是 TD 高 bias / SGD 不穩的真實樣子——和 README 的對照一致。
        probs = self.actor.forward(states)          # (T, 2)
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(T), actions] = 1.0
        grad_logits = -(one_hot - probs) * advantage.reshape(-1, 1)
        grad_logits /= N    # 除以軌跡數（和 MC 版相同）
        self.actor.backward(grad_logits, self.actor_lr)

        # 清空
        self._states      = []
        self._actions     = []
        self._rewards     = []
        self._next_states = []
        self._dones       = []


# ─────────────────────────────────────────────────────────
#  訓練迴圈
# ─────────────────────────────────────────────────────────

def train():
    env   = gym.make("CartPole-v1")
    agent = ACTDAgent()

    EPISODES  = 1000
    N_UPDATES = 4     # 每收集幾條軌跡才更新一次（和 MC 版相同）
    scores    = []

    print("=" * 60)
    print("  RL Hello World 4b — Actor-Critic (A2C) TD 版")
    print("=" * 60)
    print("\nTD Advantage：δ = r + γV(s') - V(s)")
    print("對比 MC    ：A = G_t - V(s)")
    print("差異       ：把 G_t 換成 r + γV(s')，其餘不變")
    print(f"更新時機   ：每收集 {N_UPDATES} 條軌跡後更新（和 MC 版相同）")
    print("目標       ：近 50 回合平均分 ≥ 195 = 解決！")
    print(f"\n開始訓練 {EPISODES} 個 episodes...\n")

    for episode in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0

        while True:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 比 MC 版多傳 next_state 和 done
            agent.store(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state

            if done:
                break

        scores.append(total_reward)

        # 每收集 N_UPDATES 條軌跡才更新一次（和 MC 版相同）
        if (episode + 1) % N_UPDATES == 0:
            agent.update()

        if (episode + 1) % 50 == 0:
            avg_score = np.mean(scores[-50:])
            solved = "✓ 解決！" if avg_score >= 195 else ""
            print(f"Episode {episode+1:4d} | "
                  f"近50回合平均分: {avg_score:6.1f}  {solved}")

    env.close()

    # ─────────────────────────────────────────────────────
    #  展示
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
    #  對比總結：MC vs TD
    # ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  A2C MC vs TD 對比")
    print("=" * 60)
    rows = [
        ("Critic target", "G_t（整段 return）",      "r + γV(s')（一步 bootstrap）"),
        ("Advantage",     "G_t - V(s)",               "r + γV(s') - V(s) = δ"),
        ("需要等 episode", "是",                       "否（每步可算）"),
        ("方差",           "高",                       "低"),
        ("Bias",           "無",                       "有（V 不準就偏）"),
        ("store 多存",     "無",                       "next_state, done"),
    ]
    print(f"  {'':16s} {'MC 版':28s} {'TD 版':28s}")
    print("  " + "-" * 74)
    for label, mc, td in rows:
        print(f"  {label:16s} {mc:28s} {td:28s}")

    print("\n核心洞見：")
    print("  唯一改動：G_t  →  r + γV(s')")
    print("  PPO 用的也是 TD，從這裡進 PPO 是最自然的路徑。")

    return agent


if __name__ == "__main__":
    agent = train()
    demo(lambda state: int(np.argmax(agent.actor.predict_probs(state))))
