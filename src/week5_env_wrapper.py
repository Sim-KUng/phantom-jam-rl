import gymnasium as gym
import highway_env
import pandas as pd
import numpy as np

class PhantomJamRewardWrapper(gym.Wrapper):
    """
    4주 차의 Logging 기능에 더하여, 
    5주 차 목표인 '다중 목표 보상 함수(Multi-Objective Reward)'를 계산하는 Wrapper
    """
    def __init__(self, env, log_filename="logs/phantom_jam_log_week5.csv"):
        super().__init__(env)
        self.log_filename = log_filename
        self.log_data = []
        self.step_count = 0
        
        # 보상 가중치 (Weights) 세팅
        self.w_speed = 1.0
        self.w_col = 5.0
        self.w_lane = 0.2
        self.w_global = 1.5

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.step_count = 0
        self.log_data = [] 
        return obs, info

    def step(self, action):
        # 1. 환경 기본 step 실행 (여기서 나오는 기본 reward는 무시하고 새로 계산합니다)
        obs, default_reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        
        # 2. 커스텀 Reward 계산 로직 호출
        custom_reward = self._calculate_custom_reward(action)

        # 에피소드 종료 조건 처리 및 로깅
        if terminated or truncated:
            self._save_log()

        # 기본 reward 대신 우리가 만든 custom_reward를 반환!
        return obs, custom_reward, terminated, truncated, info

    def _calculate_custom_reward(self, action):
        ego_vehicle = self.env.unwrapped.controlled_vehicles[0]
        vehicles = self.env.unwrapped.road.vehicles
        
        target_speed = 25.0 # 약 90km/h
        
        # [1] 기초 보상: 충돌 페널티
        r_collision = -1.0 if ego_vehicle.crashed else 0.0
        
        # [1] 기초 보상: 속도 유지 (0 ~ 1 사이로 정규화)
        # 충돌하지 않았을 때만 속도 보상을 줌
        r_speed = 0.0
        if not ego_vehicle.crashed:
            r_speed = min(ego_vehicle.speed / target_speed, 1.0)
            
        # [2] 얌체 운전 방지: 차선 변경 페널티 (Action 0: Left, Action 2: Right)
        r_lane_change = -1.0 if action in [0, 2] else 0.0
        
        # [3] 유령 정체 해소 (Global Reward): 에이전트 후방 차량들의 평균 속도
        r_global = 0.0
        rear_vehicles_speed = []
        
        for v in vehicles:
            if v is not ego_vehicle:
                # 에이전트보다 뒤에 있고 (x 좌표가 작음), 너무 멀지 않은(예: 100m 이내) 차량 필터링
                distance = ego_vehicle.position[0] - v.position[0]
                if 0 < distance < 100:
                    rear_vehicles_speed.append(v.speed)
                    
        if rear_vehicles_speed:
            avg_rear_speed = sum(rear_vehicles_speed) / len(rear_vehicles_speed)
            r_global = min(avg_rear_speed / target_speed, 1.0)

        # 최종 보상 합산
        total_reward = (
            self.w_speed * r_speed + 
            self.w_col * r_collision + 
            self.w_lane * r_lane_change + 
            self.w_global * r_global
        )
        
        return total_reward

    def _save_log(self):
        if self.log_data:
            import os
            os.makedirs("logs", exist_ok=True)
            df = pd.DataFrame(self.log_data)
            df.to_csv(self.log_filename, index=False)
            print(f"✅ [Logging] 데이터 저장 완료: {self.log_filename}")

# 환경 생성 함수
def create_phantom_jam_env():
    env = gym.make("highway-v0", render_mode="human")
    env.unwrapped.configure({
        "lanes_count": 2,               
        "vehicles_count": 60,           
        "vehicles_density": 2.0,        
        "controlled_vehicles": 1,       
        "duration": 50,                 
        "observation": {
            "type": "Kinematics",
            "features": ["presence", "x", "y", "vx", "vy"],
            "vehicles_count": 15,
            "normalize": True,
            "absolute": False 
        },
        "action": {
            "type": "DiscreteMetaAction",
        }
    })
    
    # 우리가 만든 Custom Reward Wrapper 적용
    return PhantomJamRewardWrapper(env)

if __name__ == "__main__":
    print("5주 차 커스텀 보상 함수 테스트를 시작합니다...")
    env = create_phantom_jam_env()
    obs, info = env.reset()
    
    # 5스텝만 진행하면서 로그 출력
    for step in range(5):
        action = env.action_space.sample() 
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {step+1} | Action: {action} | Custom Reward: {reward:.3f}")
        
        if terminated or truncated:
            print("💥 충돌 발생으로 에피소드 종료!")
            break
            
    env.close()