import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Lambda
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

class Actor(object):

	def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate):
		self.sess = sess
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.action_bound = action_bound
		self.learning_rate = learning_rate

		#표준편차의 최소값 최대값 설정
		self.std_bound = [1e-2, 1.0]

		#엑터 신경망 생성
		self.model, self.theta, self.states = self.build_network()

		#손실 함수와 그래디언트
		self.actions = tf.placeholder(tf.float32, [None, self.action_dim])
		self.advantages = tf.placeholder(tf.float32, [None, 1])

		mu_a, std_a = self.model.output
		log_policy_pdf = self.log_pdf(mu_a, std_a, self.actions) ## action차원의 하나의 벡터 형성

		loss_policy = log_policy_pdf * self.advantages #log(pi(u|x))*A(x,u)
		loss = tf.reduce_sum(-loss_policy)

		dj_dtheta = tf.gradients(loss, self.theta) ## self.theta는 trainable variable// 변수가 loss에 미치는 영향력 (기울기)
		grads = zip(dj_dtheta, self.theta) ## tf.compute_gradients 를 사용하면 기울기와 변수가 동시에 나온다 (참고)

		#변수와 기울기를 계산했으므로 apply_gradients를 적용(loss를 이용하려면 minimize 하면 됨)
		self.actor_optimizer = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)


	## Actor 신경망
	def build_network(self):
		state_input = Input((self.state_dim, ))
		h1 = Dense(64, activation = 'relu')(state_input)
		h2 = Dense(32, activation = 'relu')(h1)
		h3 = Dense(16, activation = 'relu')(h2)
		##output 출력 2개
		out_mu = Dense(self.action_dim, activation = 'tanh')(h3)
		std_output = Dense(self.action_dim, activation = 'softplus')(h3) # softplus: log(exp(x)+1)

		##평균값을 [-action_bound, action_bound] 범위로 조정
		## output 되는 mu는 tanh를 통과하기에 [-1,1] 범위, 하지만 action 범위는 -2,2 이므로 lambda 를 이용해 변환 
		mu_output = Lambda(lambda x: x*self.action_bound)(out_mu) 
		model = Model(state_input, [mu_output, std_output])
		model.summary()
		return model, model.trainable_weights, state_input

	## 로그-정책 확률밀도 함수
	def log_pdf(self, mu, std, action):
		std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
		var = std**2
		## 각 액션 차원 각각 policy의 확률 계산 후, reduce sum(연속 공간이므로 가우시안 가정)
		log_policy_pdf = -.5 * (action - mu) ** 2 / var -.5 * tf.log(var*2*np.pi)
		return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

	## 액터 신경망 출력에서 확률적으로 행동 추출
	def get_action(self, state):
		mu_a, std_a = self.model.predict(np.reshape(state, [1, self.state_dim])) # state reshape
		mu_a = mu_a[0]
		std_a = std_a[0]
		std_a = np.clip(std_a, self.std_bound[0], self.std_bound[1])
		action = np.random.normal(mu_a, std_a, size=self.action_dim)
		return action

	## 액터 신경망에서 평균값 계산
	def predict(self, state):
		mu_a, _ = self.model.predict(np.reshape(state, [1, self.state_dim]))
		return mu_a[0]

	## 액터 신경망 학습 
	def train(self, states, actions, advantages):
		self.sess.run(self.actor_optimizer, feed_dict = {self.states: states, self.actions:actions, self.advantages: advantages})

	## 액터 신경망 파라미터 저장
	def save_weights(self, path):
		self.model.save_weights(path)

	## 액터 신경망 파라미터 로드
	def load_weigths(self, path):
		self.model.load_weigths(path+'pendulum_actor.h5')


