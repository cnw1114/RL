##A3C 에이저트를 학습하고 평가
import gym
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras.backend as K
import matplotlib.pyplot as plt
import threading
import multiprocessing

from a3c_actor import Global_Actor, Worker_Actor
from a3c_critic import Global_Critic, Worker_Critic

# 모든 워커에서 공통으로 사용할 글로벌 변수 설정
global_episode_count, global_step, global_episode_reward = 0, 0, [] #총 에피소드 카운트, 총 스텝카운트, 결과저장

class A3Cagent(object):
	'''
	글로벌 신경망 생성
	'''
	def __init__(self, env_name):
		#텐서플로우 세션 설정
		self.sess = tf.InteractiveSession()
		K.set_session(self.sess)

		#학습할 환경 설정
		self.env_name = env_name
		self.WORKERS_NUM = multiprocessing.cpu_count() #워커의 개수 (내 노트북 기준 8개)
		env = gym.make(self.env_name)

		#상태변수 차원(dimension)
		state_dim = env.observation_space.shape[0]

		#행동 차원(dimension)
		action_dim = env.action_space.shape[0]

		#행동의 최대 크기
		action_bound = env.action_space.high[0]

		#글로벌 액터 및 크리틱 신경망 생성
		self.global_actor = Global_Actor(state_dim, action_dim, action_bound)
		self.global_critic = Global_Critic(state_dim)

	def train(self, max_episode_num):
		workers = []
		# A3Cworker 클래스를 이용한 워커 스레드 생성하고 리스트에 추가
		for i in range(self.WORKERS_NUM):
			worker_name = f'worker{i}'
			workers.append(A3Cworker(worker_name, self.env_name, self.sess, self.global_actor
									, self.global_critic, max_episode_num))

		#그래디언트 계산을 위한 세션 초기화
		self.sess.run(tf.global_variables_initializer())

		#리스트를 순회하면서 각 워커 스레드를 시작하고 다시 리스트를 순회하면서 조인 
		for worker in workers:
			worker.start()

		for worker in workers:
			worker.join()

		## 바로 하나의 포문에서 start(), join()을 같이사용하면 join과 동시에 thread 가 종료됨. (병렬 처리를 위해)
		
		# 학습이 끝난 후, 글로벌 누적 보상값 저장
		np.savetxt('./save_weights/pendulum_epi_reward.txt', global_episode_reward)
		print(global_episode_reward)

	## 에피소드와 글로벌 누적 보상값을 그려주는 함수
	def plot_result(self):
		plt.plot(global_episode_reward)
		plt.show()

class A3Cworker(threading.Thread):
	'''
	워커 스레드 생성
	'''
	def __init__(self, worker_name, env_name, sess, global_actor, global_critic, max_episode_num):
		threading.Thread.__init__(self)

		#하이퍼 파라미터
		self.GAMMA = .95
		self.ACTOR_LEARING_RATE = .0001
		self.CRITIC_LEARNING_RATE = .001
		self.ENTROPY_BETA = .01
		self.t_MAX = 4 ## 시간차 스텝

		self.max_episode_num = max_episode_num

		#워커의 환경 생성
		self.env = gym.make(env_name)
		self.worker_name = worker_name
		self.sess = sess

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
		self.worker_actor = Worker_Actor(self.sess, self.state_dim, self.action_dim, 
										 self.action_bound, self.ACTOR_LEARING_RATE,
										 self.ENTROPY_BETA, self.global_actor)
		self.worker_critic = Worker_Critic(self.sess, self.state_dim, self.action_dim,
										   self.CRITIC_LEARNING_RATE, self.global_critic)

		#글로벌 신경망의 파라미터를 워커 신경망으로 복사
		self.worker_actor.model.set_weights(self.global_actor.model.get_weights())
		self.worker_critic.model.set_weights(self.global_critic.model.get_weights())

	##n-step 시간차 타깃 계산
	def n_step_td_target(self, rewards, next_v_value, done):
		td_targets = np.zero_like(rewards)
		cumulative = 0
		if not done:
			cumulative = next_v_value

		for k in reversed(range(0, len(rewards))): # trajectory 의 시간만큼 t=0,...,len(reward)(=T)
			cumulative = self.GAMMA * cumulative + rewards[k]
			td_targets[k] = cumulative
		return td_targets #시간별로 시간차 타깃 계산 (벡터)

	## 배치에 저장된 데이터 추출
	def unpack_batch(self, batch):
		unpack = batch[0]
		for idx in range(len(batch)-1):
			unpack = np.append(unpack, batch[idx + 1], axis=0)
		return unpack

	def run(self): ## Thread를 상속받기에 super class method명이 run이라서 run으로 이름을 통일해주어야 한다.
		#모든 워커에서 공통으로 사용할 글로벌 변수 선언 
		global global_episode_count, global_step, global_episode_reward

		#워커 실행 시 프린트 
		print(self.worker_name, 'starts ---')
		#에피소드마다 다음을 반복
		while global_episode_count <= int(self.max_episode_num):

			#배치 초기화
			batch_state, batch_action, batch_reward = [], [], []
			#에피소드 초기화
			step, episode_reward, done = 0, 0, False
			#환경 초기화 및 초기 상태 관측
			state = self.env.reset()
			#에피소드 종료 시까지 다음을 반복

			while not done:

				#환경 가시화
				# self.env.render #멀티 스레드방식이라 창이 여러개 뜨기 때문에 주석처리
				#행동 추출
				action = self.worker_actor.get_action(state)
				#행동 범위 클리핑
				action = np.clip(action, -self.action_bound, self.action_bound)
				#다음 상태, 보상 관측
				next_state, reward, done, _ = self.env.step(action)

				#shape 변환
				state = np.reshape(state, [1, self.state_dim])
				reward = np.reshape(reward, [1,1])
				action = np.reshape(action, [1, self.action_dim])

				#batch에 저장
				batch_state.append(state)
				batch_action.append(action)
				batch_reward.append((reward+8)/8) #보상 범위 조절[-16, 0] ==> [-1, 1]

				#상태 업데이트
				state = next_state
				step += 1
				episode_reward += reward[0]

				#배치가 채워지면, 워커 학습 시작!
				if len(batch_state) == self.t_MAX or done:

					#배치에서 데이터 추출
					states = self.unpack_batch(batch_state)
					actions = self.unpack_batch(batch_action)
					rewards = self.unpack_batch(batch_reward)
					
					#배치 비우기
					batch_state, batch_action, batch_reward = [], [], []
					
					#n-step TD타깃 어드벤티지 계산
					next_state = np.reshape(next_state, [1, self.state_dim])
					next_v_value = self.worker_critic.model.predict(next_state) # 1차원 스칼라 가치값 도출
					n_step_td_targets = n_step_td_target(rewards, next_v_value, done)
					v_values = self.worker_critic.model.predict(states) # 각 시점별로 vector형성 
					advantages = n_step_td_targets - v_values

					#글로벌 크리틱과 액터 신경망 업데이트
					self.worker_critic.train(states, n_step_td_targets)
					self.worker_actor.train(states, actions, advantages)

					#글로벌 신경망 파라미터를 워커 신경망으로 복사
					self.worker_actor.model.set_weights(self.global_actor.model.get_weights())
					self.worker_critic.model.set_weights(self.global_critic.model.get_weights())

					#글로벌 스텝 업데이트
					global_step += 1
				
				#에피소드 종료
				if done:
					#글로벌 에피소드 카운트 업데이트
					global_episode_count += 1
					#episode마다 결과 보상값 출력
					print('Worker name:', self.worker_name, 
						', Episode:', global_episode_count, ', Step:', step, ', Reward:', episode_reward)
					#10번째 에피소드마다 신경망 파라미터를 파일에 저장
					if global_episode_count % 10 == 0:
						self.global_actor.save_weights('./save_weights/pendulum_actor.h5')
						self.global_critic.save_weights('./save_weights/pendulum_critic.h5')