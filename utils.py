"""
共用工具函式
"""

import gymnasium as gym


def demo(predict_fn, episodes=None):
    """
    開視窗展示訓練完的 policy

    predict_fn：接受 state，回傳 action 的函式
    episodes  ：跑幾個 episode 後自動結束（None = 無限迴圈，Ctrl+C 結束）
    """
    print("\n開啟視覺化視窗，按 Ctrl+C 結束...")
    env = gym.make("CartPole-v1", render_mode="human")
    ep  = 0
    try:
        while episodes is None or ep < episodes:
            state, _ = env.reset()
            steps = 0
            while True:
                action = predict_fn(state)
                state, _, terminated, truncated, _ = env.step(action)
                steps += 1
                if terminated or truncated:
                    print(f"  Episode {ep+1}：{steps} 步")
                    break
            ep += 1
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
