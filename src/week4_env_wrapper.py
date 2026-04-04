import gymnasium as gym
import highway_env
import pandas as pd
import numpy as np
import os


# ---------------------------------------------------------
# 1. 시공간 다이어그램용 CSV 로깅 Wrapper 클래스 구축
# ---------------------------------------------------------
class PhantomJamLoggingWrapper(gym.Wrapper):
    """
    매 스텝마다 도로 위 모든 차량의 궤적 및 속도 데이터를 추출하여
    에피소드 종료 시 CSV로 자동 저장하는 Wrapper 클래스
    """

    def __init__(self, env, log_filename="logs/phantom_jam_log_week4.csv"):
        super().__init__(env)
        self.log_filename = log_filename
        self.log_data = []
        self.step_count = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.step_count = 0
        self.log_data = []  # 에피소드 시작 시 로그 초기화
        self._record_state()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        self._record_state()

        # 에피소드가 끝나면 자동으로 CSV 파일 저장
        if terminated or truncated:
            self._save_log()

        return obs, reward, terminated, truncated, info

    def _record_state(self):
        # 모든 차량의 현재 상태 기록
        for vehicle in self.env.unwrapped.road.vehicles:
            self.log_data.append(
                {
                    "step": self.step_count,
                    "vehicle_id": id(vehicle),
                    "x_position": vehicle.position[0],
                    "y_position": vehicle.position[1],
                    "speed": vehicle.speed,
                    "lane_index": vehicle.lane_index[2],
                }
            )

    def _save_log(self):
        if self.log_data:
            os.makedirs("logs", exist_ok=True)
            df = pd.DataFrame(self.log_data)
            df.to_csv(self.log_filename, index=False)
            print(
                f"✅ [LoggingWrapper] 총 {self.step_count} 스텝의 궤적 데이터가 '{self.log_filename}'에 저장되었습니다."
            )


# ---------------------------------------------------------
# 2. State & Action이 정의된 커스텀 환경 생성 함수
# ---------------------------------------------------------
def create_phantom_jam_env():
    env = gym.make("highway-v0", render_mode="human")

    env.unwrapped.configure(
        {
            "lanes_count": 2,
            "vehicles_count": 60,
            "vehicles_density": 2.0,
            "controlled_vehicles": 1,
            "duration": 50,
            # 🎯 [State] 에이전트 주변 차량 2D 행렬 추출 (Kinematic Observation)
            "observation": {
                "type": "Kinematic",
                # 관측할 피처: 존재 여부, x좌표, y좌표, x축 속도, y축 속도
                "features": ["presence", "x", "y", "vx", "vy"],
                # 에이전트 자신을 포함해 주변에서 가장 가까운 N대의 차량 관측
                "vehicles_count": 15,
                # 에이전트 기준 상대 좌표 및 속도로 정규화할지 여부
                "normalize": True,
                "absolute": False,
            },
            # 🎯 [Action] 5가지 이산형 변수 (Discrete Meta Action)
            # 0: 좌측 차선 변경, 1: 현재 상태 유지, 2: 우측 차선 변경, 3: 가속, 4: 감속
            "action": {
                "type": "DiscreteMetaAction",
            },
        }
    )

    # 래퍼(Wrapper)를 환경에 덧씌우기
    wrapped_env = PhantomJamLoggingWrapper(env)
    return wrapped_env


if __name__ == "__main__":
    print("4주 차 환경 디버깅을 시작합니다...")
    env = create_phantom_jam_env()

    # 환경 초기화 (State 통신 테스트)
    obs, info = env.reset()
    print(f"✅ Observation Space 형태 (State): {obs.shape}")
    print(f"✅ Action Space 형태: {env.action_space}")

    # 에이전트에게 5가지 행동 중 무작위 행동을 시키며 10스텝만 테스트
    for _ in range(10):
        # env.action_space.sample()은 0~4 사이의 랜덤 정수를 반환
        random_action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(random_action)

        if terminated or truncated:
            break

    env.close()
    print("디버깅 완료! State와 Action이 Gym 인터페이스와 정상 통신합니다.")
