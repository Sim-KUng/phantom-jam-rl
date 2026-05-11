import time
from stable_baselines3 import PPO

from week10_env_wrapper import create_phantom_jam_env


def main():
    print("[week10] 학습된 PPO 모델 테스트를 시작합니다.")

    env = create_phantom_jam_env(
        render_mode="human",
        enable_logging=True,
        log_filename="logs/week10_test_log.csv",
    )

    model_path = "models/ppo_2lane_week10_final"
    model = PPO.load(model_path)

    obs, info = env.reset()

    for step in range(1000):
        action, _states = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        time.sleep(0.03)

        if terminated or truncated:
            print(f"[week10] episode ended at step {step}")
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()