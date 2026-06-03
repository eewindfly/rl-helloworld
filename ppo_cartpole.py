"""
RL Hello World 5 — PPO (PPO-Clip) on CartPole
================================================
和 actor_critic_td.py 的關係：PPO = A2C(TD) + 兩個核心改動。

【一句話總結 PPO】

  A2C 的問題：on-policy，一批資料更新一次就丟掉，樣本效率低；
              而且 advantage 一大，policy 一步就可能跨太遠 → 崩。

  PPO 的解法（唯一兩個核心）：
    1. 用「新舊策略的機率比 ratio」+「clip」把每步更新幅度夾住，
       於是同一批資料可以安全地反覆更新好幾個 epoch。
    2. 同一批 rollout 多 epoch + minibatch 複用 → 樣本效率大增。

  其餘東西（GAE、entropy bonus、共享網路、advantage 正規化…）都是
  「標配但非核心」的技巧，本檔為了聚焦核心，全部不放。
  advantage 沿用 actor_critic_td.py 的「單步 TD δ」，不用 GAE。

【相比 actor_critic_td.py 的 diff（只有這些）】

  1. update() 開頭先用「當前 actor」算一次 old_log_prob 並凍結
     （這就是 π_old，收資料的那個策略）。
  2. advantage 與 TD target 也在更新前用「舊 critic」算一次、凍結，
     整個 K-epoch 過程中保持不變（PPO 標準做法）。
  3. 把 A2C 的目標  log π · A
     換成 PPO 的 clipped surrogate：
        ratio      = π_new(a|s) / π_old(a|s)
        L^CLIP     = min( ratio · A,  clip(ratio, 1-ε, 1+ε) · A )
  4. 外層多了「K 個 epoch × minibatch」迴圈，重複用同一批資料更新。

【clip 在做什麼？（PPO 的靈魂）】

  ratio = 1 表示新策略和舊策略一樣。clip 把 ratio 限制在 [1-ε, 1+ε]。
  直覺：advantage 是正的（這動作好）→ 想提高它的機率 → ratio 往上，
        但 clip 不准 ratio 超過 1+ε，避免「一次更新衝太遠」。
        advantage 是負的（這動作差）→ ratio 往下，同理夾在 1-ε。
  取 min 的巧妙：只在「會讓目標變好」的方向限制；若新策略已經
  變差（往錯方向跑），min 會選未截斷項，讓梯度照常把它拉回來。
"""

import numpy as np
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

        self.gamma   = 0.99
        # PPO 一次收一批資料要反覆更新 K_EPOCHS 次，等於同一批資料
        # 被「重用 K 次」，所以單次 lr 要比 A2C(TD) 的 0.02 小一點，
        # 否則累積更新過頭。critic 仍要夠快學準（bootstrap 靠它）。
        self.actor_lr  = 0.003
        self.critic_lr = 0.01

        # ── PPO 核心超參 ──
        self.clip_eps     = 0.2   # ε：ratio 被夾在 [1-ε, 1+ε]
        self.k_epochs     = 4     # 同一批資料重複更新幾個 epoch
        self.minibatch_sz = 64    # 每個 minibatch 的 transition 數

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

    def update(self):
        states      = np.array(self._states)       # (T, 4)
        actions     = np.array(self._actions)      # (T,)
        rewards     = np.array(self._rewards)       # (T,)
        next_states = np.array(self._next_states)  # (T, 4)
        terminateds = np.array(self._terminateds)  # (T,) bool 真終止
        T = len(rewards)

        # ══════════════════════════════════════════════════════════
        #  Phase 1：更新前，用「舊策略 / 舊 critic」算好並凍結
        #           （以下三樣在整個 K-epoch 過程中都不變）
        # ══════════════════════════════════════════════════════════

        # (a) old_log_prob = log π_old(a|s)
        #     此刻的 actor 就是收集這批資料的策略 π_old。
        #     .copy() 凍結，後續 actor 更新不會動到它。
        old_probs_all   = self.actor.forward(states)            # (T, 2)
        old_log_probs   = np.log(old_probs_all[np.arange(T), actions] + 1e-10).copy()

        # (b) TD target = r + γV(s')，只有真終止才把 V(s') 歸零
        #     （truncated 照常 bootstrap —— 和 TD 版完全相同的處理）
        next_values = self.critic.forward(next_states)          # (T,)
        next_values[terminateds] = 0.0
        td_targets = (rewards + self.gamma * next_values).copy()  # (T,)

        # (c) advantage = TD target - V_old(s)，單步 TD δ（不用 GAE）
        old_values = self.critic.forward(states)                # (T,)
        advantages = (td_targets - old_values).copy()           # (T,)

        # ══════════════════════════════════════════════════════════
        #  Phase 2：同一批資料，跑 K 個 epoch，每 epoch 切 minibatch
        # ══════════════════════════════════════════════════════════
        for _ in range(self.k_epochs):
            idx = np.random.permutation(T)        # 每個 epoch 重新洗牌
            for start in range(0, T, self.minibatch_sz):
                mb = idx[start:start + self.minibatch_sz]
                mb_states  = states[mb]
                mb_actions = actions[mb]
                mb_adv     = advantages[mb]
                mb_oldlogp = old_log_probs[mb]
                mb_target  = td_targets[mb]
                m = len(mb)

                # ── 更新 Actor：clipped surrogate ──────────────
                probs = self.actor.forward(mb_states)           # (m, 2)
                new_logp = np.log(probs[np.arange(m), mb_actions] + 1e-10)

                ratio     = np.exp(new_logp - mb_oldlogp)       # π_new / π_old
                unclipped = ratio * mb_adv
                clipped   = np.clip(ratio, 1 - self.clip_eps,
                                           1 + self.clip_eps) * mb_adv

                # L^CLIP = min(unclipped, clipped)，我們要最大化它。
                # 梯度只在「min 選到未截斷項」時流動；選到截斷項且
                # ratio 已飽和時，clip 的導數為 0 → 該樣本不更新。
                # 用 mask 表示：unclipped <= clipped 時取 ratio 的梯度。
                use_grad = (unclipped <= clipped).astype(np.float64)   # (m,)

                # ∇log π_a 對 logits = (one_hot - probs)
                # ∇ratio = ratio · ∇log π_a
                # 最大化 → loss = -L^CLIP → grad_logits 帶負號
                one_hot = np.zeros_like(probs)
                one_hot[np.arange(m), mb_actions] = 1.0
                coef = -(use_grad * ratio * mb_adv) / m         # (m,) 每筆係數
                grad_logits = coef.reshape(-1, 1) * (one_hot - probs)
                self.actor.backward(grad_logits, self.actor_lr)

                # ── 更新 Critic：value MSE（target 已凍結）──────
                v = self.critic.forward(mb_states)              # (m,)
                grad_v = (v - mb_target) / m
                self.critic.backward(grad_v, self.critic_lr)

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

    EPISODES   = 1000
    UPDATE_EVERY = 8     # 每收集 8 條軌跡才更新一次（PPO 喜歡大一點的 batch）
    scores     = []

    print("=" * 60)
    print("  RL Hello World 5 — PPO (PPO-Clip)")
    print("=" * 60)
    print(f"\nclip ε = {agent.clip_eps}   K_epochs = {agent.k_epochs}   "
          f"minibatch = {agent.minibatch_sz}")
    print("核心   ：ratio + clip + 同批多 epoch 複用")
    print("advantage：單步 TD δ（沿用 A2C-TD，不用 GAE）")
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

        if ep_in_batch == UPDATE_EVERY:
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
    print("  A2C(TD) vs PPO")
    print("=" * 60)
    rows = [
        ("Actor 目標",   "log π · A",                 "min(ratio·A, clip(ratio)·A)"),
        ("資料複用",      "用一次就丟",                 "同批 K epoch 反覆用"),
        ("更新幅度控制",  "靠調 lr",                    "靠 clip 夾住 ratio"),
        ("advantage",    "單步 TD δ",                  "單步 TD δ（相同）"),
        ("樣本效率",      "低",                         "高"),
    ]
    print(f"  {'':14s} {'A2C(TD)':28s} {'PPO':30s}")
    print("  " + "-" * 74)
    for label, a, p in rows:
        print(f"  {label:14s} {a:28s} {p:30s}")
    print("\n核心洞見：")
    print("  PPO = A2C + 「ratio + clip」+「同批多 epoch 複用」")
    print("  RLHF（ChatGPT 的訓練方式）用的就是 PPO。")

    return agent


if __name__ == "__main__":
    agent = train()
    demo(lambda state: int(np.argmax(agent.actor.predict_probs(state))))
