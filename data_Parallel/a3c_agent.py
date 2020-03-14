import gym
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras.backend as K
import matplotlib.pyplot as plt
import threading
import multiprocessing

from a3c_actor import Global_Actor, Worker_Actor
from a3c_critic import Global_Critic, Worker_Critic

#모든 워커에서 공통으로 사용할 글로벌 변수 설정
global_episode_count = 0 #총 에피소드 카운트
global_step = 0#총 스텝 카운트
global_episode_reward = []#결과 저장

graph = tf.get_default_graph()

class A3Cagent(object):
	'''
	글로벌 신경망 생성
	'''
	def __init__(self, env_name):
		##텐서플로우 세션 설정
		self.sess = tf.Session()
		K.set_session(self.sess)

		#학습할 환경 설정
		self.env_name = env_name
		self.WORKERS_NUM = multiprocessing.cpu_count()#워커의 개수 설정

		#하이퍼파라미터
		self.ACTOR_LEARNING_RATE = .0001
		self.CRITIC_LEARNING_RATE = .001
		self.ENTROPY_BETA = .01

		#상태 변수 차원(dimension)
		env = gym.make(self.env_name)
		state_dim = env.observation_space.shape[0]
		#행동 차원(dimension)
		action_dim = env.action_space.shape[0]
		#행동의 최대 크기
		action_bound = env.action_space.high[0]
		#글로벌 액터 및 크리틱 신경망 생성
		self.global_actor = Global_Actor(self.sess, state_dim, action_dim, action_bound
			, self.ACTOR_LEARNING_RATE, self.ENTROPY_BETA)
		self.global_critic = Global_Critic(self.sess, state_dim, action_dim, self.CRITIC_LEARNING_RATE)

		#그래디언트 계산을 위한 세션 초기화
		self.sess.run(tf.global_variables_initializer())

	def train(self, max_episode_num):
		workers = []
		# A3Cworker 클래스를 이용한 워커 스레드를 생성하고 리스트에 추가
		for i in range(self.WORKERS_NUM):
			worker_name = f'worker{i}'
			workers.append(A3Cworker(worker_name, self.env_name, self.global_actor, self.global_critic, max_episode_num))

		#리스트를 순회하면서 각 워커 스레드를 시작하고 다시 리스트를 순회하면서 조인
		for worker in workers:
			worker.start()
		for worker in workers:
			worker.join()

		#학습이 끝난 후, 글로벌 누적 보상값 저장
		np.savetxt('./save_weights/pendulum_epi_reward.txt', global_episode_reward)
		print(global_episode_reward)

	##에피소드와 글로벌 누적 보상값을 그려주는 함수
	def plot_result(self):
		plt.plot(global_episode_reward)
		plt.show()

class A3Cworker(threading.Thread):
	'''
	워커 스레드 생성
	'''
	def __init__(self, worker_name, env_name, global_actor, global_critic, max_episode_num):
		super().__init__()

		##하이퍼 파라미터
		self.GAMMA = .95
		self.t_MAX = 4 #n-step 시간차

		self.max_episode_num = max_episode_num

		#워커의 환경 생성
		self.env = gym.make(env_name)
		self.worker_name = worker_name

		#글로벌 신경망 공유
		self.global_actor = global_actor
		self.global_critic = global_critic

		#상태변수 차원(dimension)
		self.state_dim = self.env.observation_space.shape[0]
		#행동 차원(dimension)
		self.action_dim = self.env.action_space.shape[0]
		#행동의 최대 크기
		self.action_bound = self.env.action_space.high[0]

		#워커 액터 및 크리틱 신경망 생성
		self.worker_actor = Worker_Actor(self.state_dim, self.action_dim, self.action_bound)
		self.worker_critic = Worker_Critic(self.state_dim)
		#글로벌 신경망의 파라미터를 워커 신경망으로 복사
		self.worker_actor.model.set_weights(self.global_actor.model.get_weights()) ## np.array형태의 파라미터를 적용하기 위해 set_weights
		self.worker_critic.model.set_weights(self.global_critic.model.get_weights())

	#n-step 시간차 타깃 계산
	def n_step_td_target(self, rewards, next_v_value, done):
		td_targets = np.zeros_like(rewards)
		cumulative = 0
		if not done:
			cumulative = next_v_value

		for k in reversed(range(0, len(rewards))):
			cumulative = self.GAMMA*cumulative + rewards[k]
			td_targets[k] = cumulative
		return td_targets

	#배치에 저장된 데이터 추출
	def unpack_batch(self, batch):
		unpack = batch[0]
		for idx in range(len(batch)-1):
			unpack = np.append(unpack, batch[idx+1], axis=0)
		return unpack

	#파이썬에서 스레드를 구동하기 위해서는 함수명을 run으로 해주어야함. 워커의 학습을 구현
	def run(self):
		#모든 워커에서 공통으로 사용할 글로벌 변수 선언
		global global_episode_count, global_step, global_episode_reward
		#워커 실행시 프린트
		print(self.worker_name, 'starts ---')
		#에피소드 마다 다음을 반복
		while global_episode_count <= int(self.max_episode_num):
			#배치 초기화
			batch_state, batch_action, batch_reward = [], [], []
			#에피소드 초기화
			step, episode_reward, done = 0, 0, False
			#환경 초기화 및 초기 상태 관측
			state = self.env.reset()
			#에피소드 종료 시까지 다음을 반복
			while not done:
				#행동 추출
				action = self.worker_actor.get_action(state)
				#행동 범위 클리핑
				action = np.clip(action, -self.action_bound, self.action_bound)
				#다음 상태, 보상 관측
				next_state, reward, done, _ = self.env.step(action)

				#shape 변환
				state = np.reshape(state, [1, self.state_dim])
				reward = np.reshape(reward, [1, 1])
				action = np.reshape(action, [1, self.action_dim])

				#배치에 저장
				batch_state.append(state)
				batch_action.append(action)
				batch_reward.append((reward+8)/8)

				#상태 업데이트
				state = next_state
				step += 1
				episode_reward += reward[0]

				#배치가 채워지면, 글로벌 신경망 학습 시작
				if len(batch_state) == self.t_MAX or done:

					#배치에서 데이터 추출
					states = self.unpack_batch(batch_state)
					actions = self.unpack_batch(batch_action)
					rewards = self.unpack_batch(batch_reward)
					#배치 비움
					batch_state, batch_action, batch_reward = [], [], []
					#n-step TD 타깃과 어드밴티지 계산
					next_state = np.reshape(next_state, [1, self.state_dim])
					next_v_value = self.global_critic.model.predict(next_state)
					n_step_td_targets = self.n_step_td_target(rewards, next_v_value, done)
					
					with graph.as_default():
						v_values = self.global_critic.model.predict(states)
						
					advantages = n_step_td_targets - v_values
					#글로벌 크리틱과 액터 신경망 업데이트 
					self.global_critic.train(states, n_step_td_targets)
					self.global_actor.train(states, actions, advantages)

					#글로벌 신경망 파라미터를 워커 신경망으로 복사
					self.worker_actor.model.set_weights(self.global_actor.model.get_weights())
					self.worker_critic.model.set_weights(self.global_critic.model.get_weights())

					#글로벌 스텝 업데이트
					global_step += 1
				#에피소드가 종료
				if done:
					#글로벌 에피소드 카운트 업데이트 
					global_episode_count += 1
					#에피소드마다 결과 보상값 출력
					print('Worker name:', self.worker_name, 'Episode:', global_episode_count, 'step:', step, 'Reward', episode_reward)
					global_episode_reward.append(episode_reward)

					if global_episode_count % 10 == 0:
						self.global_actor.save_weights('./save_weights/pendulum_actor.h5')
						self.global_critic.save_weights('./save_weights/pendulum_critic.h5')
