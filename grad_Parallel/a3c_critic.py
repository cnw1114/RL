from keras.models import Model
from keras.layers import Dense, Input
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

## 크리틱 신경망 
def build_network(state_dim):
	state_input = Input((state_dim,))
	h1 = Dense(64, activation='relu')(state_input)
	h2 = Dense(32, activation='relu')(h1)
	h3 = Dense(16, activation='relu')(h2)
	v_output = Dense(1, activation='linear')(h3)
	model = Model(state_input, v_output)
	# model.summary()
	model._make_predict_function()
	return model, model.trainable_weights, state_input

class Global_Critic(object):
	'''
	글로벌 크리틱 신경망, 파라미터만 필요하므로 학습기능 없음
	'''
	def __init__(self, state_dim):
		self.state_dim = state_dim
		#크리틱 신경망 생성
		self.model, self.phi, _ = build_network(state_dim)

	## 크리틱 파라미터 저장
	def save_weights(self, path):
		self.model.save_weights(path)

	## 크리틱 신경망 파라미터 로드
	def load_weights(self, path):
		self.model.load_weights(path+'pendulum_critic.h5')

class Worker_Critic(object):
	'''
	워커 크리틱 신경망
	'''
	def __init__(self, sess, state_dim, action_dim, learning_rate, global_critic):
		self.sess = sess
		self.global_critic = global_critic
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.learning_rate = learning_rate

		#크리틱 신경망 생성
		self.model, self.phi, self.states = build_network(self.state_dim)

		#시간차 타깃을 담을 플레이스홀더
		self.td_targets = tf.placeholder(tf.float32, [None, 1])

		#워커의 손실함수와 그래디언트
		v_values = self.model.output #abstract tensor
		loss = tf.reduce_sum(tf.square(self.td_targets-v_values))
		dj_dphi = tf.gradients(loss, self.phi)

		#그래디언트 클리핑
		dj_dphi, _ = tf.clip_by_global_norm(dj_dphi, 40)

		
		
		#워커의 그래디언트를 이용해 글로벌 신경망 업데이트
		grads = zip(dj_dphi, self.global_critic.phi)
		
		self.critic_optimizer = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

	##워커 크리틱 신경망 학습
	def train(self, states, td_targets):

		self.sess.run(self.critic_optimizer, feed_dict={
					  self.states: states,
					  self.td_targets: td_targets
			})