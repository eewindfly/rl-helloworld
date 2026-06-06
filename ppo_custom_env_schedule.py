"""
RL Hello World 6a — PPO on 自訂環境：多機台排程（離散動作）
==========================================================
這是「階段 6」的第一個自訂環境。重點不再是改演算法——而是**自己定義問題**。

【階段 6 想讓你體會的事】

  前面 1~5b，每一階段的主角都是「演算法」（Q-Learning → DQN → PG → A2C →
  PPO）。環境（CartPole / Pendulum）是別人寫好的。
  真實工作裡剛好相反：**演算法你直接拿現成的 PPO，難的是把你的問題包成
  一個 Gymnasium 環境**——定義 state / action / reward，尤其是 reward shaping。
  這一階段就是要親手體驗：「定義問題」比「改演算法」更需要動腦。

【本檔的問題：線上機台排程（Online Load Balancing）】

  一條一條進來的工作（job），每個有不同大小（處理時間）。你要即時決定把
  它丟到 M 台機器中的哪一台。目標：讓最後「最忙那台機器的總負載（makespan）」
  越小越好——也就是把負載盡量平均分散。

    state  ：[目前這個 job 的大小, 各機台目前負載(去均值), 進度比例]
    action ：把 job 丟到哪一台（離散，M 選 1）
    reward ：-(這步造成的 makespan 增量)  ← 見下方 reward shaping 說明
    最佳直覺策略：丟給「目前最閒」的機台（greedy / LPT 類）

【★ 與前面階段的關係：PPO 演算法「一行不改」】

  本檔 import 階段 5 的 `PPOAgent`（ppo_cartpole.py），其 update / compute_gae /
  choose_action **全部原封不動繼承**。唯一的改動是：

    1. 換一個自訂環境（這份檔案的 SchedulingEnv，就是「定義問題」本身）。
    2. 因為觀測是 5 維、動作是 3 個（CartPole 是 4 維、2 個），所以把 Actor /
       Critic 的「輸入/輸出維度」換掉——注意這只動網路的 in/out 尺寸，
       PPO 的演算法邏輯（ratio + clip + GAE + K 次複用）完全沒碰。

  → 這正是階段 6 的精神：演算法是現成的，你出力的地方是「環境」。

【reward shaping：本檔最該細看的地方】

  makespan = max(各機台負載)。我們給的每步 reward 是：
      r_t = -(makespan_after − makespan_before)
  把整條 episode 的 r_t 加起來會「望遠鏡式」抵銷成 −(最終 makespan)：
      Σ r_t = −(makespan_final − 0) = −makespan_final
  所以它和「只在最後給 −makespan」的總獎勵完全等價，但**每一步都有訊號**
  （dense reward），PPO 學得快、critic 也好估。這就是 reward shaping 的價值：
  在不改變「最終目標」的前提下，把稀疏的終局獎勵拆成密集的逐步獎勵。

  （只有「被選中的機台變成新的最忙者」時這步才扣分 → 逼 agent 學會避開
    已經很忙的機台，自然湧現出「挑最閒的丟」的負載平衡行為。）
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch

# ── 直接沿用階段 5 的 PPO：演算法一行不改，只在下面換網路維度 ──
from ppo_cartpole import PPOAgent
from pg_cartpole import PolicyNetwork          # 離散 Actor
from actor_critic import CriticNetwork         # Critic


# ═════════════════════════════════════════════════════════════
#  1. 自訂環境：把「排程問題」包成標準 Gymnasium 介面
#     —— 這份 class 就是階段 6 真正的新東西（定義問題）
# ═════════════════════════════════════════════════════════════

class SchedulingEnv(gym.Env):
    """
    線上機台排程環境（負載平衡）。

    一個 episode = 依序進來 num_jobs 個 job，每個 job 大小隨機。
    agent 每步把「當前 job」指派給 num_machines 台中的一台。
    episode 結束（所有 job 派完）時看 makespan = max(各機台負載)，越小越好。

    Gymnasium 介面四件套：
      observation_space / action_space ：宣告 state、action 的形狀與範圍
      reset() ：開新一局，回傳第一個觀測
      step(a)：執行動作，回傳 (obs, reward, terminated, truncated, info)
    """
    metadata = {"render_modes": []}

    def __init__(self, num_machines=3, num_jobs=20,
                 job_low=0.1, job_high=1.0, seed=None):
        super().__init__()
        self.num_machines = num_machines
        self.num_jobs     = num_jobs
        self.job_low      = job_low
        self.job_high     = job_high

        # 觀測 = [當前 job 大小, 各機台負載(去均值) × M, 進度比例] → 維度 M+2
        obs_dim = num_machines + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        # 動作 = 選哪一台機器（離散 M 選 1）
        self.action_space = spaces.Discrete(num_machines)

        self._rng = np.random.default_rng(seed)

    # ── 產生一局的所有 job 大小，並重置機台負載 ──
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.loads = np.zeros(self.num_machines, dtype=np.float64)
        self.jobs  = self._rng.uniform(self.job_low, self.job_high, size=self.num_jobs)
        self.idx   = 0                       # 還沒派的第一個 job 的索引
        return self._obs(), {}

    # ── 把當前狀態組成觀測向量 ──
    def _obs(self):
        cur_job = self.jobs[self.idx] if self.idx < self.num_jobs else 0.0
        rel_loads = self.loads - self.loads.mean()   # 去均值 → 表示「相對忙閒」，數值有界
        frac_done = self.idx / self.num_jobs
        return np.concatenate(
            [[cur_job], rel_loads, [frac_done]]).astype(np.float32)

    # ── 執行一步：把當前 job 丟給機台 action ──
    def step(self, action):
        makespan_before = self.loads.max()
        self.loads[action] += self.jobs[self.idx]    # 指派：該機台負載增加
        makespan_after = self.loads.max()

        # reward shaping：這步造成的 makespan 增量（見檔頭說明，總和 = −最終 makespan）
        reward = -(makespan_after - makespan_before)

        self.idx += 1
        truncated  = (self.idx >= self.num_jobs)     # 所有 job 派完 → 這局結束
        terminated = False                            # 沒有「中途失敗」這種事
        return self._obs(), reward, terminated, truncated, {"makespan": self.loads.max()}


# ═════════════════════════════════════════════════════════════
#  2. PPO Agent：繼承階段 5 的 PPOAgent，只換網路維度
#     —— 演算法（update/compute_gae/choose_action）全部繼承，零修改
# ═════════════════════════════════════════════════════════════

class SchedulingPPOAgent(PPOAgent):
    """
    和 ppo_cartpole.PPOAgent 的唯一差別：Actor 輸出維度 = 機台數、
    Actor/Critic 輸入維度 = 觀測維度。其餘（GAE、ratio、clip、K 次複用、
    所有超參）完全繼承不動。
    """
    def __init__(self, obs_dim, num_actions):
        super().__init__()                            # 先建好預設網路 + 超參 + optimizer
        # 只換掉網路（輸入/輸出維度配合本環境），演算法邏輯不碰
        self.actor  = PolicyNetwork(input_dim=obs_dim, output_dim=num_actions)
        self.critic = CriticNetwork(input_dim=obs_dim)
        # 換了網路 → optimizer 要重綁到新參數（lr 沿用父類別的 0.001 / 0.005）
        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=self.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)


# ═════════════════════════════════════════════════════════════
#  3. 對照用的簡單啟發式策略（給「學得好不好」一把尺）
# ═════════════════════════════════════════════════════════════

def _rollout_makespan(env, policy_fn, episodes=200, seed=0):
    """跑若干局，回傳平均 makespan。policy_fn: (env, obs) → action"""
    rng_seed = seed
    total = 0.0
    for _ in range(episodes):
        obs, _ = env.reset(seed=rng_seed); rng_seed += 1
        while True:
            a = policy_fn(env, obs)
            obs, _, term, trunc, info = env.step(a)
            if term or trunc:
                total += info["makespan"]; break
    return total / episodes


def greedy_policy(env, obs):
    """啟發式上界參考：丟給目前最閒（負載最小）的機台。"""
    return int(np.argmin(env.loads))


def random_policy(env, obs):
    """隨機亂丟，當作下界參考。"""
    return env.action_space.sample()


# ═════════════════════════════════════════════════════════════
#  4. 訓練迴圈（和 ppo_cartpole.py 的 train 幾乎一樣，只是換了 env / agent）
# ═════════════════════════════════════════════════════════════

def train():
    NUM_MACHINES = 3
    NUM_JOBS     = 20
    env   = SchedulingEnv(num_machines=NUM_MACHINES, num_jobs=NUM_JOBS, seed=0)
    obs_dim = env.observation_space.shape[0]
    agent = SchedulingPPOAgent(obs_dim=obs_dim, num_actions=NUM_MACHINES)

    BATCH_EPISODES = 4
    NUM_UPDATES    = 300     # 比 CartPole(250) 略多；排程稍難，純訓練預算
    EPISODES       = BATCH_EPISODES * NUM_UPDATES
    scores         = []

    print("=" * 64)
    print("  RL Hello World 6a — PPO 自訂環境：多機台排程（離散）[PyTorch]")
    print("=" * 64)
    print(f"\n機台數 = {NUM_MACHINES}   每局 job 數 = {NUM_JOBS}")
    print("演算法 ：完全繼承階段 5 的 PPO（ratio + clip + GAE + K 次複用），一行不改")
    print("新東西 ：SchedulingEnv（定義問題）+ reward shaping（每步 makespan 增量）")
    print(f"clip ε = {agent.clip_eps}   K_epochs = {agent.k_epochs}   GAE λ = {agent.gae_lambda}")

    # 訓練前先量「隨機」與「貪婪」兩個參考點
    ref_env = SchedulingEnv(num_machines=NUM_MACHINES, num_jobs=NUM_JOBS)
    ms_random = _rollout_makespan(ref_env, random_policy, episodes=200, seed=10_000)
    ms_greedy = _rollout_makespan(ref_env, greedy_policy, episodes=200, seed=10_000)
    print(f"\n參考點（平均 makespan，越小越好）：隨機 ≈ {ms_random:.3f} ，貪婪(最閒優先) ≈ {ms_greedy:.3f}")
    print(f"目標   ：訓練後 PPO 的 makespan 逼近貪婪 ≈ {ms_greedy:.3f}")
    print(f"\n開始訓練 {EPISODES} 個 episodes...\n")

    ep_in_batch = 0
    for episode in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0.0
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

        if (episode + 1) % 100 == 0:
            # total_reward = −makespan，取負就是平均 makespan
            avg_ms = -np.mean(scores[-50:])
            tag = "✓ 逼近貪婪！" if avg_ms <= ms_greedy * 1.03 else ""
            print(f"Episode {episode+1:4d} | 近50局平均 makespan: {avg_ms:6.3f}  {tag}")

    env.close()

    # ── 展示：用學到的確定性 policy（argmax）量 makespan ──
    print("\n" + "=" * 64)
    print("  訓練完成！評估學到的 policy（200 局平均 makespan）")
    print("=" * 64)
    eval_env = SchedulingEnv(num_machines=NUM_MACHINES, num_jobs=NUM_JOBS)
    def learned_policy(env, obs):
        return int(np.argmax(agent.actor.predict_probs(obs)))
    ms_learned = _rollout_makespan(eval_env, learned_policy, episodes=200, seed=10_000)

    print(f"\n  隨機亂丟        ：makespan ≈ {ms_random:.3f}")
    print(f"  PPO 學到的策略  ：makespan ≈ {ms_learned:.3f}")
    print(f"  貪婪(最閒優先)  ：makespan ≈ {ms_greedy:.3f}  ← 啟發式參考上界")
    gap = (ms_learned - ms_greedy) / ms_greedy * 100
    print(f"\n  PPO 比貪婪差 {gap:+.1f}%（越接近 0 越好；負代表還更好）")

    print("\n核心洞見：")
    print("  1. PPO 演算法一行沒改——本階段的工作量全在『定義環境 + reward shaping』。")
    print("  2. 我們沒教它『挑最閒的』；它從『每步 makespan 增量』這個 reward 自己學出")
    print("     負載平衡行為。reward 設計對了，行為就湧現了。")
    return agent


if __name__ == "__main__":
    train()
