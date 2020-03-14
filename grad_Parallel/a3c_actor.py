import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Lambda
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

##액터 신경망
def build_network(state_dim, action_dim, action_bound):
	state_input = Input((state_dim, ))
	h1 = Dense(64, activation='relu')(state_input)
	h2 = Dense(32, activation='relu')(h1)
	h3 = Dense(16, activation='relu')(h2)
	out_mu = Dense(action_dim, activation='tanh')(h3)
	std_output = Dense(action_dim, activation='softplus')(h3)

	#평균값을 [-action_bound, action_bound] 범위로 조정
	mu_output = Lambda(lambda x: x*action_bound)(out_mu) # tanh값 -1,1의 범위이므로 -2,2의 action값의 scale을 맞추기 위해
	model = Model(state_input, [mu_output, std_output])
	# model.summary()
	model._make_predict_function() #class 안에서 def가 정의 되면 필요없음
	return model, model.trainable_weights, state_input

class Global_Actor(object):
	'''
	글로벌 액터 신경망. 
	파라미터만 필요하므로 학습 기능은 없음
	'''
	def __init__(self, state_dim, action_dim, action_bound):
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.action_bound = action_bound
		#글로벌 액터 신경망 생성
		self.model, self.theta, self.states = build_network(self.state_dim, self.action_dim, self.action_bound)

	##액터 신경망에서 평균값을 계산
	def predict(self,state):
		mu_a, _ = self.model.predict(np.reshape(state, [1, self.state_dim]))
		return mu_a[0]
	
	##액터 신경망 파라미터 저장
	def save_weights(self, path):
		self.model.save_weights(path)

	##액터 신경망 파라미터 로드
	def load_weights(self, path):
		self.model.load_weights(path + 'pendulum_actor.h5')

class Worker_Actor(object):
	'''
	워커 액터 신경망
	'''
	def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, entropy_beta, global_actor):
		self.sess = sess
		self.global_actor = global_actor

		self.state_dim = state_dim
		self.action_dim = action_dim
		self.action_bound = action_bound
		self.learning_rate = learning_rate

		# 표준편차의 최대값 최소값 설정
		self.std_bound = [1e-2, 1]
		
		#워커의 액터 신경망 생성
		self.model, self.theta, self.states = build_network(self.state_dim, self.action_dim, self.action_bound)

		#정책과 어드벤티지를 담을 플레이스 홀더
		self.actions = tf.placeholder(tf.float32, [None, self.action_dim])
		self.advantages = tf.placeholder(tf.float32, [None, 1])

		#정책 확률밀도함수 및 엔트로피
		mu_a, std_a = self.model.output
		log_policy_pdf, entropy = self.log_pdf(mu_a, std_a, self.actions)

		#워커의 손실함수와 그래디언트
		loss_policy = log_policy_pdf * self.advantages
		loss = tf.reduce_sum(-loss_policy-entropy_beta*entropy)
		dj_dtheta = tf.gradients(loss, self.theta)

		#그래디언트 글리핑
		dj_dtheta, _ = tf.clip_by_global_norm(dj_dtheta, 40)

		#워커의 그래디언트를 이용해 글로벌 신경망 업데이트
		grads = zip(dj_dtheta, self.global_actor.theta)
		self.actor_optimizer = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

	#로그-정책 확률밀도함수 및 엔트로피 계산
	def log_pdf(self, mu, std, action):
		std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
		var = std**2
		log_policy_pdf = -.5 * (action-mu)**2 / var - .5 * tf.log(var*2*np.pi)
		entropy = .5 * (tf.log(2*np.pi*std**2) + 1.0) #1/2*(1+log(2pi*var(x))) <=-integral(pi()logpi())꼴이므로
		return tf.reduce_sum(log_policy_pdf, 1, keepdims=True), tf.reduce_sum(entropy, 1, keepdims=True)

	#액터 신경망 출력에서 확률적으로 행동을 추출
	def get_action(self, state):
		mu_a, std_a = self.model.predict(np.reshape(state, [1,self.state_dim]))
		mu_a = mu_a[0]
		std_a = std_a[0]
		std_a = np.clip(std_a, self.std_bound[0], self.std_bound[1])
		action = np.random.normal(mu_a, std_a, size=self.action_dim)
		return action

	#워커 액터 신경망 학습
	def train(self, states, actions, advantages):

		self.sess.run(self.actor_optimizer, feed_dict={
					  self.states: states,
					  self.actions: actions,
					  self.advantages: advantages
			})








