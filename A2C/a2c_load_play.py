import gym
from a2c_agent import A2Cagent

def main():
	env_name = 'Pendulum-v0'
	env = gym.make(env_name) # 환경으로 openAI Gym의 pendulum-v0 설정
	agent = A2Cagent(env) # A2C 에이전트 객체

	agent.actor.load_weights('./save_weights/') #액터 신경망 파라미터 로드
	agent.critic.load_weights('./save_weights') #크리틱 신경망 파라미터 로드

	time = 0
	state = env.reset() # 환경 초기화 및 초기상태 관측

	while True:
		env.render()
		action = agent.actor.predict(state) # 행동 계산
		state, reward, done, _ = env.step(action) #환경으로부터 다음 상태, 보상 받음
		time += 1

		print('Time:',time, 'Reward:',reward)
		if done:
			break
	env.close()

if __name__ == '__main__':
	main()


