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

        self.gamma = 0.99
        # PPO 一次收一批資料要反覆更新 K_EPOCHS 次，等於同一批資料
        # 被「重用 K 次」，所以單次 lr 要比 A2C(TD) 小一點，否則累積更新過頭。
        self.actor_lr  = 0.0015
        self.critic_lr = 0.005

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=self.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

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
        states      = torch.as_tensor(np.array(self._states),      dtype=torch.float32)  # (T,4)
        actions     = torch.as_tensor(np.array(self._actions),     dtype=torch.long)     # (T,)
        rewards     = torch.as_tensor(np.array(self._rewards),     dtype=torch.float32)  # (T,)
        next_states = torch.as_tensor(np.array(self._next_states), dtype=torch.float32)  # (T,4)
        terminateds = torch.as_tensor(np.array(self._terminateds), dtype=torch.bool)     # (T,)
        T = len(self._rewards)

        # ══════════════════════════════════════════════════════════
        #  Phase 1：更新前，用「舊策略 / 舊 critic」算好並凍結
        #           （no_grad → 這三樣在整個 K-epoch 過程中都是常數）
        # ══════════════════════════════════════════════════════════
        with torch.no_grad():
            # (a) old_log_prob = log π_old(a|s)：此刻的 actor 就是 π_old
            old_logits    = self.actor(states)                       # (T, 2)
            old_log_probs = Categorical(logits=old_logits).log_prob(actions)  # (T,)

            # (b) TD target = r + γV(s')，只有真終止才把 V(s') 歸零
            #     （truncated 照常 bootstrap —— 和 TD 版完全相同的處理）
            next_values = self.critic(next_states)                   # (T,)
            next_values[terminateds] = 0.0
            td_targets = rewards + self.gamma * next_values          # (T,)

            # (c) advantage = TD target - V_old(s)，單步 TD δ（不用 GAE）
            old_values = self.critic(states)                         # (T,)
            advantages = td_targets - old_values                     # (T,)

        # ══════════════════════════════════════════════════════════
        #  Phase 2：同一批資料，跑 K 個 epoch，每 epoch 切 minibatch
        # ══════════════════════════════════════════════════════════
        for _ in range(self.k_epochs):
            idx = torch.randperm(T)               # 每個 epoch 重新洗牌
            for start in range(0, T, self.minibatch_sz):
                mb = idx[start:start + self.minibatch_sz]
                mb_states  = states[mb]
                mb_actions = actions[mb]
                mb_adv     = advantages[mb]
                mb_oldlogp = old_log_probs[mb]
                mb_target  = td_targets[mb]

                # ── 更新 Actor：clipped surrogate ──────────────
                logits   = self.actor(mb_states)                     # (m, 2)
                new_logp = Categorical(logits=logits).log_prob(mb_actions)  # (m,)

                ratio = torch.exp(new_logp - mb_oldlogp)             # π_new / π_old
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps,
                                           1 + self.clip_eps) * mb_adv

                # L^CLIP = min(surr1, surr2)，最大化它 → loss 取負。
                # 「被 clip 那側、ratio 飽和則不更新」的梯度遮罩，
                # 由 torch.min + torch.clamp 的次梯度自動完成（手刻版的 use_grad）。
                actor_loss = -torch.min(surr1, surr2).mean()
                self.actor_opt.zero_grad()
                actor_loss.backward()
                self.actor_opt.step()

                # ── 更新 Critic：value MSE（target 已凍結）──────
                v = self.critic(mb_states)                           # (m,)
                critic_loss = F.mse_loss(v, mb_target)
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

    BATCH_EPISODES = 8     # 每收集 8 條軌跡才更新一次（batch size，PPO 喜歡大一點的 batch）
    NUM_UPDATES    = 125   # 總共要更新幾次（真正決定學不學得起來的量）
    EPISODES       = BATCH_EPISODES * NUM_UPDATES
    scores         = []

    print("=" * 60)
    print("  RL Hello World 5 — PPO (PPO-Clip) [PyTorch]")
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
