# src/test.py
import time
from stable_baselines3 import PPO
from week5_env_wrapper import create_phantom_jam_env


def main():
    print("저장된 PPO 모델 테스트를 시작합니다...")

    # 1. 환경 생성
    env = create_phantom_jam_env()

    # 2. 학습된 모델 불러오기
    model_path = "models/ppo_phantom_jam_test"
    model = PPO.load(model_path)

    # 3. 테스트 루프
    obs, info = env.reset()
    for i in range(500):
        # 모델이 현재 상태(obs)를 보고 최적의 행동(action)을 예측
        action, _states = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        # 화면이 너무 빨리 지나가지 않도록 약간의 딜레이 추가
        time.sleep(0.05)

        if terminated or truncated:
            print(f"💥 {i} 스텝 만에 에피소드 종료 (충돌 또는 시간 초과)")
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
