import gym
from a2c_agent import A2Cagent 

def main():
	max_episode_num = 1000 #최대 에피소드 설정
	env_name = 'Pendulum-v0'
	env = gym.make(env_name) # 환경으로 OpenAI Gym의 Pendulum-v0 설정
	agent = A2Cagent(env)

	# 학습 진행
	agent.train(max_episode_num)

	# 학습 결과 도시
	agent.plot_result()

if __name__ == "__main__":
	main()
