import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from week10_env_wrapper import create_phantom_jam_env


def main():
    print("[week11] 2차 튜닝 PPO 추가 학습을 시작합니다.")

    os.makedirs("logs/tensorboard", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # 2차 튜닝 보상 가중치
    reward_config = {
        "w_speed": 1.3,
        "w_col": 6.0,
        "w_lane": 0.15,
        "w_global": 2.2,
        "target_speed": 25.0,
    }

    env = create_phantom_jam_env(
        render_mode=None,
        enable_logging=False,
        reward_config=reward_config,
    )

    env = Monitor(env)

    # 기존 30,000 step 체크포인트에서 이어서 학습
    model_path = "checkpoints/ppo_2lane_week10_30000_steps"

    model = PPO.load(
        model_path,
        env=env,
        tensorboard_log="./logs/tensorboard/",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path="./checkpoints/",
        name_prefix="ppo_2lane_week11_tuned",
    )

    model.learn(
        total_timesteps=200_000,
        tb_log_name="ppo_2lane_week11_tuned",
        callback=checkpoint_callback,
        reset_num_timesteps=False,
        progress_bar=True,
    )

    final_model_path = "models/ppo_2lane_week11_tuned_final"
    model.save(final_model_path)

    env.close()

    print(f"[week11] 2차 튜닝 학습 완료: {final_model_path}.zip")


if __name__ == "__main__":
    main()