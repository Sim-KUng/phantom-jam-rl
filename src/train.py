import os
import gymnasium as gym
from stable_baselines3 import PPO

# 5주 차에 만든 커스텀 환경 생성 함수 가져오기
from week5_env_wrapper import create_phantom_jam_env


def main():
    print("6주 차 PPO 모델 연동 및 초기 학습 테스트를 시작합니다.")

    # 1. 환경 세팅
    env = create_phantom_jam_env()

    # 2. 신경망(MLP) 네트워크 구조 정의
    # (15, 5) 크기의 행렬을 Stable-Baselines3가 자동으로 Flatten(평탄화)하여 입력으로 받습니다.
    # 복잡한 다차선 주변 상황을 파악해야 하므로 은닉층 노드 수를 [256, 256]으로 넉넉하게 잡습니다.
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

    # 3. PPO 모델 초기 하이퍼파라미터 세팅
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=512,  # 빠른 시범 구동을 위해 업데이트 주기를 짧게 설정 (기본 2048)
        batch_size=64,
        gamma=0.99,  # 미래 보상 할인율
        policy_kwargs=policy_kwargs,
        verbose=1,  # 터미널에 학습 진행 로그(ep_rew_mean 등) 출력
        tensorboard_log="./logs/tensorboard/",  # 텐서보드 로그 저장 경로
    )

    # 4. 소규모 에피소드 실행 (1000 스텝)
    print("\n본격적인 학습 루프(1000 steps)를 가동합니다...")
    model.learn(total_timesteps=1000, progress_bar=True)

    # 5. 모델 저장
    os.makedirs("models", exist_ok=True)
    model_path = "models/ppo_phantom_jam_test"
    model.save(model_path)

    print(
        f"\n✅ 초기 학습 테스트 완료! 모델이 '{model_path}.zip'에 정상적으로 저장되었습니다."
    )
    env.close()


if __name__ == "__main__":
    main()
