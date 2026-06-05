"""
RL Hello World 5 — PPO (PPO-Clip) on CartPole
================================================
和 actor_critic_gae.py 的關係：PPO = A2C(GAE) + 兩個核心改動。
（advantage 沿用階段 4c 的 GAE——這才是真實 PPO 的標準配置。）

【一句話總結 PPO】

  A2C 的問題：on-policy，一批資料更新一次就丟掉，樣本效率低；
              而且 advantage 一大，policy 一步就可能跨太遠 → 崩。

  PPO 的解法（唯一兩個核心）：
    1. 用「新舊策略的機率比 ratio」+「clip」把每步更新幅度夾住，
       於是同一批資料可以安全地反覆更新好幾個 epoch。
    2. 同一批 rollout 多 epoch + minibatch 複用 → 樣本效率大增。

  其餘東西（entropy bonus、共享網路、advantage 正規化…）都是
  「標配但非核心」的技巧，本檔為了聚焦核心，全部不放。
  advantage 用階段 4c 的 GAE（λ=0.95）——GAE 是 advantage 估計法，
  和 clip 完全正交，已在 4c 單獨介紹過，這裡直接沿用即可。

【相比 actor_critic_gae.py 的 diff（只有這些）】

  1. update() 開頭先用「當前 actor」算一次 old_log_prob 並凍結
     （這就是 π_old，收資料的那個策略）。
  2. advantage（GAE）與 critic target（λ-return）也在更新前用「舊 critic」
     算一次、凍結，整個 K-epoch 過程中保持不變（PPO 標準做法）。
  3. 把 A2C 的目標  log π · A
     換成 PPO 的 clipped surrogate：
        ratio      = π_new(a|s) / π_old(a|s)        ← important sampling
        L^CLIP     = min( ratio · A,  clip(ratio, 1-ε, 1+ε) · A )
  4. 外層多了「K 次 full-batch 更新」迴圈，重複用同一批資料更新。

  → 換句話說：相對 4c，advantage 的算法（GAE）一字不改，新增的純粹是
    「clip + 同批 K 次複用」。這才是 PPO 真正、唯一的概念增量。

【為什麼這版「不切 minibatch」？（最小 diff 的精神）】

  minibatch 不是 PPO 的核心，它只是 SGD 的工程細節（省記憶體、加梯度
  噪聲），AC(GAE) 一樣能切 minibatch。把它放進「PPO vs AC」的 diff 會
  混淆兩件正交的事。PPO 相對 AC(GAE) 真正且唯一必要的概念增量只有：

    (i)  important sampling：資料用 π_old 收，要重複拿來更新 π_new，
         就必須用 ratio = π_new/π_old 修正「採樣分布 ≠ 當前分布」的
         偏差——這正是「同一批資料能反覆更新」的理論依據。
    (ii) clip：純 IS 反覆更新會因 ratio 爆掉而發散；clip 把每步幅度
         夾住，讓「重複利用」變安全。

  因此本版採「最小 diff」：整批資料、做 K 次完整 update，不切 minibatch。
  （若想加 minibatch 純粹是優化技巧，與上面兩點無關。）

【clip 在做什麼？（PPO 的靈魂）】

  ratio = 1 表示新策略和舊策略一樣。clip 把 ratio 限制在 [1-ε, 1+ε]。
  直覺：advantage 是正的（這動作好）→ 想提高它的機率 → ratio 往上，
        但 clip 不准 ratio 超過 1+ε，避免「一次更新衝太遠」。
        advantage 是負的（這動作差）→ ratio 往下，同理夾在 1-ε。
  取 min 的巧妙：只在「會讓目標變好」的方向限制；若新策略已經
  變差（往錯方向跑），min 會選未截斷項，讓梯度照常把它拉回來。

────────────────────────────────────────────────────────────
【關於這版：從 numpy 手刻 → PyTorch（PPO 受益最大）】

  手刻 numpy 版最容易出錯的就是 clip 的「梯度遮罩」：
    L^CLIP = min(unclipped, clipped) 在被 clip 那側、ratio 飽和時，
    clip 的導數是 0 → 該樣本不更新。numpy 版要自己算一個
    use_grad = (unclipped <= clipped) 的 mask，再手動套進梯度。

  PyTorch 版完全不用管這件事：
    surr1 = ratio * adv
    surr2 = clamp(ratio, 1-ε, 1+ε) * adv
    loss  = -min(surr1, surr2).mean()
    loss.backward()
  torch.min 和 torch.clamp 都帶正確的「次梯度」，autograd 會自動讓
  梯度只從被選中、且沒被 clamp 飽和的那一路流回去——你手刻過的 mask
  其實就是這個次梯度，現在交給 autograd。
"""

# ════════════════════════════════════════════════════════════
# 【PPO 核心公式速查表（對照 update() 內的程式碼）】
#
#   符號：θ = 新策略參數、θ_old = 收資料時凍結的舊策略、
#         ε = clip 幅度、γ = 折扣因子、V = critic（價值函數）。
#
#   (1) 機率比 ratio
#         r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
#                = exp( log π_θ(a_t|s_t) − log π_θ_old(a_t|s_t) )
#       → 程式：ratio = exp(new_logp − old_logp)
#
#   (2) GAE Advantage（沿用階段 4c，λ=0.95）
#         δ_t   = r_t + γV(s_{t+1}) − V(s_t)     （終止時 V(s_{t+1}) = 0）
#         Â_t   = Σ_{l≥0} (γλ)^l · δ_{t+l}        （反向掃描累積，見 compute_gae）
#       → 程式：advantages = compute_gae(deltas, dones)
#
#   (3) Critic target（λ-return，維持 target = adv + V 的不變式）
#         y_t = Â_t + V(s_t)
#       → 程式：returns = advantages + values
#
#   (4) Clipped surrogate（PPO 的靈魂，Actor 目標）
#         L^CLIP(θ) = E_t[ min( r_t(θ)·Â_t,
#                               clip(r_t(θ), 1−ε, 1+ε)·Â_t ) ]
#       Actor loss = −L^CLIP（目標要最大化 → 取負做梯度下降）
#       → 程式：actor_loss = −min(surr1, surr2).sum() / N
#         （除以軌跡數 N，與 AC(TD) 對齊；epoch 1 梯度 = AC(TD) 梯度）
#
#   (5) Value loss（Critic 目標，MSE；target = λ-return y_t）
#         L^VF(φ) = E_t[ ( V_φ(s_t) − y_t )² ]
#       → 程式：critic_loss = mse_loss(v, returns)
# ════════════════════════════════════════════════════════════

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
from utils import demo
from pg_cartpole import PolicyNetwork as ActorNetwork
from actor_critic import CriticNetwork   # Critic 架構完全相同，直接複用


# ─────────────────────────────────────────────────────────
#  PPO Agent
# ─────────────────────────────────────────────────────────

class PPOAgent:
    def __init__(self):
        self.actor  = ActorNetwork()
        self.critic = CriticNetwork()

        self.gamma      = 0.99
        self.gae_lambda = 0.95   # 沿用階段 4c 的 GAE λ，完全相同
        # 對齊 AC(GAE)：actor_lr=0.001、critic_lr=0.005，與其完全相同。
        # 這樣連單步步長都一致，PPO 與 AC(GAE) 的差別就純剩三項核心
        # （ratio、clip、K 次複用），沒有任何被 lr 藏起來的隱性 diff。
        self.actor_lr  = 0.001
        self.critic_lr = 0.005

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=self.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # ── PPO 核心超參 ──
        self.clip_eps = 0.2   # ε：ratio 被夾在 [1-ε, 1+ε]
        self.k_epochs = 4     # 同一批資料整批重複更新幾次（full-batch，不切 minibatch）

        # buffer：和 actor_critic_td.py 完全相同
        #   （PPO 不用額外存 old_log_prob —— 在 update() 開頭，
        #     當前 actor 就是 π_old，現算一次即可，見 update()）
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
        """反向掃描把單步 δ 累積成 GAE：Â_t = δ_t + γλ·Â_{t+1}
        （和 actor_critic_gae.py 的同名函式完全相同；done 為界重置累積）"""
        T = len(deltas)
        advantages = torch.zeros(T)
        last = 0.0
        for t in reversed(range(T)):
            if dones[t]:
                last = 0.0
            last = deltas[t] + self.gamma * self.gae_lambda * last
            advantages[t] = last
        return advantages

    def update(self):
        states      = torch.as_tensor(np.array(self._states),      dtype=torch.float32)  # (T,4)
        actions     = torch.as_tensor(np.array(self._actions),     dtype=torch.long)     # (T,)
        rewards     = torch.as_tensor(np.array(self._rewards),     dtype=torch.float32)  # (T,)
        next_states = torch.as_tensor(np.array(self._next_states), dtype=torch.float32)  # (T,4)
        terminateds = torch.as_tensor(np.array(self._terminateds), dtype=torch.bool)     # (T,)
        T = len(self._rewards)
        N = int(np.sum(self._dones))   # 完整軌跡數，actor loss 除以它 → 與 AC(TD) 一致

        # ══════════════════════════════════════════════════════════
        #  Phase 1：更新前，用「舊策略 / 舊 critic」算好並凍結
        #           （no_grad → 這三樣在整個 K-epoch 過程中都是常數）
        # ══════════════════════════════════════════════════════════
        with torch.no_grad():
            # (a) old_log_prob = log π_old(a|s)：此刻的 actor 就是 π_old
            #     公式：log π_θ_old(a_t | s_t)         ← 之後算 ratio 的分母
            old_logits    = self.actor(states)                       # (T, 2)
            old_log_probs = Categorical(logits=old_logits).log_prob(actions)  # (T,)

            # (b) 每步 TD 殘差 δ：terminated 才把 V(s') 歸零，truncated 照常 bootstrap
            #     δ_t = r_t + γV(s_{t+1}) − V(s_t)
            values      = self.critic(states)                        # (T,)  V(s_t)
            next_values = self.critic(next_states)                   # (T,)  V(s_{t+1})
            next_values[terminateds] = 0.0
            deltas      = rewards + self.gamma * next_values - values  # (T,)  δ_t

            # (c) advantage = GAE（公式 2，沿用 4c）；critic target = λ-return（公式 3）
            advantages = self.compute_gae(deltas, self._dones)       # (T,)  Â_t^GAE
            returns    = advantages + values                         # (T,)  λ-return = y_t

        # ══════════════════════════════════════════════════════════
        #  Phase 2：同一批資料（整批），重複做 K 次完整 update
        #           ── 這就是「AC(GAE) + IS + clip」的最小 diff：
        #              和 AC(GAE) 唯一的差別是 (1) 目標換成 clipped
        #              surrogate、(2) 同批資料更新 K 次而非 1 次。
        #              advantage（GAE）算法完全沿用 4c，不切 minibatch。
        # ══════════════════════════════════════════════════════════
        for _ in range(self.k_epochs):
            # ── 更新 Actor：clipped surrogate（整批）──────────
            # 公式：log π_θ(a_t | s_t)  ← ratio 的分子（用「當前」θ 算）
            logits   = self.actor(states)                            # (T, 2)
            new_logp = Categorical(logits=logits).log_prob(actions)  # (T,)

            # 公式 (1)：r_t(θ) = exp( log π_θ − log π_θ_old )
            #   AC(GAE) 用的是 log π_θ 本身；PPO 改用「比值」做 important
            #   sampling，才能合法地拿 π_old 收的資料反覆更新 π_new。
            ratio = torch.exp(new_logp - old_log_probs)              # r_t(θ) = π_new/π_old
            # 公式 (4) 兩項：surr1 = r_t·Â_t ，surr2 = clip(r_t,1−ε,1+ε)·Â_t
            surr1 = ratio * advantages                               # r_t(θ)·Â_t
            surr2 = torch.clamp(ratio, 1 - self.clip_eps,
                                       1 + self.clip_eps) * advantages  # clip(r_t,1−ε,1+ε)·Â_t

            # 公式 (4)：L^CLIP = E[ min(surr1, surr2) ]，最大化它 → loss 取負。
            #   actor_loss = −L^CLIP
            # 「被 clip 那側、ratio 飽和則不更新」的梯度遮罩，
            # 由 torch.min + torch.clamp 的次梯度自動完成（手刻版的 use_grad）。
            #
            # ⚠️ 除以 N（軌跡數）而非 .mean()（除 T）：與 AC(GAE) 的
            #    actor_loss = -(log π·A).sum()/N 對齊。這讓「epoch 1」的
            #    梯度精確等於 AC(GAE)：ratio=1 時 min(...)=A 且 ∇(ratio·A)=∇log π·A，
            #    normalization 也一致 → PPO 成為 AC(GAE) 的乾淨超集。
            #    （T/N = 平均 episode 長度且會隨訓練變動；用 mean 會把它藏進 lr。
            #     註：Adam 會把整體常數縮放大半吸收，故 lr 不必大改。）
            actor_loss = -torch.min(surr1, surr2).sum() / N         # −L^CLIP(θ)，/N 同 AC(TD)
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # ── 更新 Critic：value MSE（target = λ-return，已凍結，整批）──
            # 公式 (5)：L^VF(φ) = E[ ( V_φ(s_t) − y_t )² ]
            v = self.critic(states)                                  # (T,)  V_φ(s_t)
            critic_loss = F.mse_loss(v, returns)                    # (V_φ(s_t) − y_t)²
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

        # 清空 buffer
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
    agent = PPOAgent()

    # ⚠️ 對齊 AC(GAE)：batch=4、共 250 次更新、總計 1000 episodes，完全相同。
    #    batch 大小是正交超參（AC 也能調大），非 PPO 核心，故對齊以保持最小 diff。
    #    （PPO 實務上偏好更大的 batch 讓 IS 重複利用更穩，那是「標配但非核心」，
    #      和 entropy bonus 同一類，本檔一律不加。clip 已能讓小 batch 安全複用。）
    BATCH_EPISODES = 4     # 每收集 4 條軌跡才更新一次（batch size，與 AC(TD) 相同）
    NUM_UPDATES    = 250   # 總共要更新幾次（與 AC(TD) 相同，總環境互動量一致）
    EPISODES       = BATCH_EPISODES * NUM_UPDATES
    scores         = []

    print("=" * 60)
    print("  RL Hello World 5 — PPO (PPO-Clip) [PyTorch]")
    print("=" * 60)
    print(f"\nclip ε = {agent.clip_eps}   K_epochs = {agent.k_epochs}   "
          f"(full-batch，不切 minibatch)")
    print("核心   ：ratio(important sampling) + clip + 同批 K 次複用")
    print(f"advantage：GAE λ={agent.gae_lambda}（沿用階段 4c，真 PPO 的標配）")
    print("目標   ：近 50 回合平均分 ≥ 195 = 解決！")
    print(f"\n開始訓練 {EPISODES} 個 episodes...\n")

    ep_in_batch = 0
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
        ep_in_batch += 1

        if ep_in_batch == BATCH_EPISODES:
            agent.update()
            ep_in_batch = 0

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

    # ── 對比總結：A2C(TD) vs PPO ──
    print("\n" + "=" * 60)
    print("  A2C(GAE) vs PPO")
    print("=" * 60)
    rows = [
        ("Actor 目標",   "log π · A",                 "min(ratio·A, clip(ratio)·A)"),
        ("資料複用",      "用一次就丟",                 "同批 K epoch 反覆用"),
        ("更新幅度控制",  "靠調 lr",                    "靠 clip 夾住 ratio"),
        ("advantage",    "GAE (λ=0.95)",              "GAE (λ=0.95，相同)"),
        ("樣本效率",      "低",                         "高"),
    ]
    print(f"  {'':14s} {'A2C(GAE)':28s} {'PPO':30s}")
    print("  " + "-" * 74)
    for label, a, p in rows:
        print(f"  {label:14s} {a:28s} {p:30s}")
    print("\n核心洞見：")
    print("  PPO = A2C(GAE) + 「ratio + clip」+「同批多 epoch 複用」")
    print("  advantage 用 GAE（4c 那塊拼圖）——這才是真實 PPO 的標準配置。")
    print("  RLHF（ChatGPT 的訓練方式）用的就是 PPO。")

    return agent


if __name__ == "__main__":
    agent = train()
    demo(lambda state: int(np.argmax(agent.actor.predict_probs(state))))
