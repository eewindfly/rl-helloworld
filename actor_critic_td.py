"""
RL Hello World 4b — Actor-Critic (A2C) TD 版 on CartPole
=========================================================
和 actor_critic.py 的唯一「概念」差異：把 MC return 換成 TD target。
（⚠️ 另有一處非概念差異：lr 也重調了，actor 0.0005→0.001、critic
  0.001→0.005，見下方 __init__ 註解。lr 是整個 pg→AC→PPO 系列唯一
  沒被固定的對照變數——batch / 更新次數 / 正規化 /N / gamma / 網路都對齊，
  唯獨 lr 因與演算法尺度耦合而每階段重調，屬必要重調而非正交超參。）

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

────────────────────────────────────────────────────────────
【關於這版：從 numpy 手刻 → PyTorch】

  和 MC 版一樣，只有梯度實作換成 autograd。TD 版另有一個 PyTorch
  要特別小心的點：

    bootstrap 的 V(s') 是「目標」，不能對它回傳梯度，否則 critic 會
    去優化「讓自己的預測等於自己」這種退化目標。所以 td_target 整段
    用 with torch.no_grad() 算、再 detach。V(s)（被訓練的那個）才留梯度。

  （手刻 numpy 版沒這個坑，因為它本來就只對 V(s) 那一路寫了 backward。）
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
#  Actor-Critic TD Agent
# ─────────────────────────────────────────────────────────

class ACTDAgent:
    def __init__(self):
        self.actor  = ActorNetwork()
        self.critic = CriticNetwork()

        self.gamma = 0.99
        # ⚠️ 學習率仍要比 MC 版動一下（原因見下方 update() 長註解）：
        #   critic 要學快一點，TD 完全靠 critic 算 advantage，critic 不準
        #   δ 連符號都不對。actor 則維持小步，配合 K 後面 PPO 的延續。
        self.actor_lr  = 0.001
        self.critic_lr = 0.005

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=self.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

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
        states      = torch.as_tensor(np.array(self._states),      dtype=torch.float32)  # (T,4)
        actions     = torch.as_tensor(np.array(self._actions),     dtype=torch.long)     # (T,)
        rewards     = torch.as_tensor(np.array(self._rewards),     dtype=torch.float32)  # (T,)
        next_states = torch.as_tensor(np.array(self._next_states), dtype=torch.float32)  # (T,4)
        terminateds = torch.as_tensor(np.array(self._terminateds), dtype=torch.bool)     # (T,)

        N = int(np.sum(self._dones))   # 完整軌跡數（和 MC 版的 N 語意相同）

        # ── 步驟 1：算 TD target（這是「目標」，不回傳梯度 → no_grad）──
        #
        #   TD target = r + γ × V(s')
        #
        #   ⚠️ V(s') 只在「真終止 terminated」時才歸零，不是 done！
        #      - terminated（桿子倒了）：s' 真的沒有未來 → V(s')=0 正確。
        #      - truncated（撐到 500 步上限被截斷）：桿子還立著，s' 還有未來價值，
        #        必須照常用 V(s') bootstrap。若也歸零，等於告訴 critic
        #        「撐到滿分的狀態價值是 0」，反而懲罰最好的軌跡 → 造成回檔。
        #
        with torch.no_grad():
            next_values = self.critic(next_states)           # (T,)
            next_values[terminateds] = 0.0                   # 只有真終止才歸零
            td_targets = rewards + self.gamma * next_values  # (T,)

        # ── 步驟 2：Critic 估 V(s)（這個要留梯度，critic 要學它）──
        values = self.critic(states)                         # (T,)

        # ── 步驟 3：TD Advantage（= TD error δ）──────────────
        #
        #   δ_t = r_t + γ V(s') - V(s) = TD target - V(s)
        #   對比 MC：Advantage = G_t - V(s)，只是把 G_t 換成 TD target。
        #   .detach()：advantage 只當 Actor 權重，不讓梯度流回 Critic。
        #
        advantage = (td_targets - values).detach()           # (T,)

        # ── 步驟 4：更新 Critic（讓 V(s) → TD target，MSE）──
        critic_loss = F.mse_loss(values, td_targets)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ── 步驟 5：更新 Actor（和 MC 版完全相同，只是 advantage 換成 δ）──
        #
        # ══════════════════════════════════════════════════════════════════
        #  ⚠️ 公式和 MC 版一字不差，為什麼 TD 版還是比較難調、容易抖？
        # ══════════════════════════════════════════════════════════════════
        #
        #    MC advantage：A = G_t          - V(s)
        #    TD advantage：δ = r + γV(s')   - V(s)
        #
        #  兩者「期望值其實一樣」（E[G_t]=E[r+γV(s')]=Q(s,a)），CartPole 的
        #  真實 advantage 本來就很小。差別在「單筆樣本」與「初期方向」：
        #
        #  1. 初期 critic 還沒學好（V≈0）時，兩者行為天差地遠：
        #       MC ≈ G_t - 0 = G_t：仍帶真實 return 的資訊（mean 幾十），
        #                           就算 critic 沒用也學得動（退化成 REINFORCE）。
        #       TD ≈ r + 0 - 0 = r ≈ +1：CartPole 每步（含倒下那步）reward 都是
        #                           +1，於是「每一步」advantage 都 ≈ +1，連失敗那步
        #                           都是正的 → actor 被告知「每個動作都好」，沒方向。
        #     ⇒ 所以 critic 一定要先學準，δ 的符號才有意義（倒下那步的 δ 變大負值
        #        來懲罰）。這就是 critic_lr 要調大的原因。
        #
        #  2. bootstrap 帶 bias：δ 用 V(s') 估未來，V 不準 → 梯度方向偏；
        #     policy 一變 V 就過時，advantage 跟著退化 → 偶爾回檔。
        #     MC 的 G_t 是無 bias 的真實 return，沒這問題。
        #
        #  （補充：手刻純 SGD 版還要額外把 actor_lr 放大約 40x，因為 δ 單筆量級
        #    比 MC 的 G_t 小很多、SGD 每步 ≈ lr×δ；改用 Adam 後，Adam 會對梯度
        #    做 RMS 正規化，量級差異大半被吸收，所以這版 lr 不必放那麼誇張——
        #    但上面 1、2 兩點是演算法本質，跟用不用 Adam 無關，依然成立。）
        # ══════════════════════════════════════════════════════════════════
        logits    = self.actor(states)            # (T, 2)
        dist      = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)        # (T,)
        actor_loss = -(log_probs * advantage).sum() / N
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
    agent = ACTDAgent()

    BATCH_EPISODES = 4     # 每收集幾條軌跡才更新一次（batch size，和 MC 版相同）
    NUM_UPDATES    = 250   # 總共要更新幾次（真正決定學不學得起來的量）
    EPISODES       = BATCH_EPISODES * NUM_UPDATES
    scores         = []

    print("=" * 60)
    print("  RL Hello World 4b — Actor-Critic (A2C) TD 版 [PyTorch]")
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
