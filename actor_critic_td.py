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
    - 真終止（terminated，桿子倒了）：V(s_{t+1}) = 0
      ⚠️ 但「撐到時間上限被截斷」（truncated）不算真終止，V(s') 照常 bootstrap

【TD error = Advantage】

  δ_t = r_t + γ × V(s_{t+1}) - V(s_t)
      = TD target - V(s_t)

  這就是 TD 版的 Advantage。
  直接用 δ_t 更新 Actor：方差小，但因 V 不準而有 bias。

  MC Advantage：A_t = G_t - V(s_t)         ← 無 bias，方差高
  TD Advantage：δ_t = r_t + γV(s') - V(s)  ← 有 bias，方差低

【程式碼改動（相比 actor_critic.py）】

  1. store() 多存 next_state、terminated、done
     （terminated 給 V(s') 歸零；done 只給數軌跡數 N）
  2. 拿掉 compute_returns()（不再需要 G_t）
  3. update() 裡：
       TD target  = r + γ × V(s')，只有 terminated 時 V(s') = 0
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
        #   詳細原因見下方 update() 的長註解；這裡先講結論：
        #   - actor_lr 調大（0.0005 → 0.02，約 40x）：
        #       TD 的 δ「單筆數值」比 MC 的 (G_t - V) 小很多（見下方說明），
        #       SGD 每步更新 ≈ lr × advantage，advantage 小了就要把 lr 放大補回來，
        #       否則 actor 幾乎不動。
        #   - critic_lr 調大（0.001 → 0.05）：
        #       TD 是 bootstrap，advantage 完全靠 critic 算，critic 要夠準才有意義，
        #       必須讓 V(s) 快點學起來。
        self.actor_lr  = 0.02
        self.critic_lr = 0.05

        # 比 MC 版多存 next_state、terminated、done
        #   terminated：桿子真的倒了 → 真終止，V(s')=0
        #   done       ：episode 結束（terminated 或 truncated）→ 只用來數軌跡數 N
        #   ⚠️ 兩者要分開！truncated（撐到時間上限被截斷）算 done 但不算 terminated，
        #      它的 s' 還有未來價值，不能把 V(s') 歸零。詳見 update()。
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

    def update(self):
        states      = np.array(self._states)       # (T, 4)
        actions     = np.array(self._actions)      # (T,)
        rewards     = np.array(self._rewards)      # (T,)
        next_states = np.array(self._next_states)  # (T, 4)
        terminateds = np.array(self._terminateds)  # (T,) bool 真終止
        dones       = np.array(self._dones)        # (T,) bool episode 結束

        T = len(rewards)
        N = int(dones.sum())   # 完整軌跡數（和 MC 版的 N 語意相同）

        # ── 步驟 1：算 TD target ──────────────────────────────
        #
        #   TD target = r + γ × V(s')
        #
        #   ⚠️ V(s') 只在「真終止 terminated」時才歸零，不是 done！
        #      - terminated（桿子倒了）：s' 真的沒有未來 → V(s')=0 正確。
        #      - truncated（撐到 500 步上限被截斷）：桿子還立著，s' 還有未來價值，
        #        必須照常用 V(s') bootstrap。若也歸零，等於告訴 critic
        #        「撐到滿分的狀態價值是 0」，反而懲罰最好的軌跡 → 造成回檔。
        #
        next_values = self.critic.forward(next_states)   # (T,)
        next_values[terminateds] = 0.0                   # 只有真終止才歸零
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
        # ══════════════════════════════════════════════════════════════════
        #  ⚠️ 為何「公式一樣」，照抄 MC 版的 lr 卻學不起來？
        # ══════════════════════════════════════════════════════════════════
        #
        #  先講一個容易誤會的點：
        #
        #    MC advantage：A = G_t          - V(s)
        #    TD advantage：δ = r + γV(s')   - V(s)
        #
        #  兩者「期望值其實一樣」！因為
        #       E[G_t | s,a] = E[r + γV(s') | s,a] = Q(s,a)
        #    ⇒ 兩個 advantage 的期望都是同一個真實 advantage  A(s,a)=Q(s,a)-V(s)。
        #  而 CartPole 的真實 advantage 本來就很小（換個動作對未來影響不大）。
        #  所以「理論上不該差太多」這個直覺是對的——指的是期望。
        #
        #  真正差很多的是「單筆樣本的大小」，也就是 variance：
        #    - MC 的 G_t 是整段未來的隨機和 → 單筆會劇烈擺盪到 ±幾十（高 variance）。
        #    - TD 的 δ 只看一步           → 單筆穩穩落在 ~1（低 variance）。
        #  SGD 每步乘的是「這個有雜訊的單筆估計」、不是期望，
        #  所以 MC 單筆大 → 小 lr 就夠；TD 單筆小 → 要把 lr 放大才補得回來。
        #
        #  還有一個關鍵，解釋為什麼 TD 不只是「訊號小」而是初期根本沒方向：
        #  上面「期望相同」只在 critic 準時才成立。訓練初期 V(s) ≈ 0：
        #    - MC ≈ G_t - 0 = G_t：仍帶著真實 return 的資訊（mean ≈ 真實價值，幾十），
        #                          就算 critic 沒用也學得動（本質退化成 REINFORCE）。
        #    - TD ≈ r + 0 - 0 = r ≈ +1：真實資訊全丟了，而且 CartPole 每一步
        #                          （含倒下那步）reward 都是 +1，於是「每一步」的
        #                          advantage 都 ≈ +1，連失敗那步都是正的
        #                          → actor 被告知「每個動作都很好」，完全沒方向
        #                          （原 lr 實測 avg 卡在 ~13）。
        #
        #  小結，兩件事疊加：
        #    1. critic_lr 調大：讓 V(s) 快點學準。critic 一準，δ 的符號才會對
        #       （倒下那步的 δ 會變成大負值來懲罰），TD 才開始有意義。
        #    2. actor_lr 調大：δ 單筆量級小，要放大 lr 才有足夠的更新力道。
        #
        #  代價：訓練會比 MC 版抖（偶爾回檔），原因有二：
        #    (a) bootstrap 帶來 bias——δ 用 V(s') 估未來，V 不準時梯度方向會偏；
        #        policy 一變 V 就過時，advantage 跟著退化而回檔。
        #        MC 的 G_t 是無 bias 的真實 return，沒這問題。
        #    (b) δ 動態範圍大——倒下那步 δ≈-35、其他步≈1，偶爾來一發大梯度就會晃。
        #  這正是 TD 高 bias 的真實樣子
        #  ——對應本檔開頭與 actor_critic.py docstring 的 MC vs TD 對照表。
        # ══════════════════════════════════════════════════════════════════
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
        self._terminateds = []
        self._dones       = []


# ─────────────────────────────────────────────────────────
#  訓練迴圈
# ─────────────────────────────────────────────────────────

def train():
    env   = gym.make("CartPole-v1")
    agent = ACTDAgent()

    BATCH_EPISODES = 4     # 每收集幾條軌跡才更新一次（batch size，和 MC 版相同）
    NUM_UPDATES    = 250   # 總共要更新幾次（真正決定學不學得起來的量）
    EPISODES       = BATCH_EPISODES * NUM_UPDATES
    scores         = []

    print("=" * 60)
    print("  RL Hello World 4b — Actor-Critic (A2C) TD 版")
    print("=" * 60)
    print("\nTD Advantage：δ = r + γV(s') - V(s)")
    print("對比 MC    ：A = G_t - V(s)")
    print("差異       ：把 G_t 換成 r + γV(s')，其餘不變")
    print(f"更新時機   ：每收集 {BATCH_EPISODES} 條軌跡後更新（和 MC 版相同），共 {NUM_UPDATES} 次")
    print("目標       ：近 50 回合平均分 ≥ 195 = 解決！")
    print(f"\n開始訓練 {EPISODES} 個 episodes...\n")

    for episode in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0

        while True:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 比 MC 版多傳 next_state、terminated、done
            #   terminated 給 V(s') 歸零用；done 只給數軌跡數用（見 store/update）
            agent.store(state, action, reward, next_state, terminated, done)
            total_reward += reward
            state = next_state

            if done:
                break

        scores.append(total_reward)

        # 每收集 BATCH_EPISODES 條軌跡才更新一次（和 MC 版相同）
        if (episode + 1) % BATCH_EPISODES == 0:
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
