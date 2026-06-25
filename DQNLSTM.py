import numpy as np
import tensorflow as tf
from keras import layers, models

class DQNLSTM:
    def __init__(self, actions, num_channel, learning_rate=0.001, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.n_actions = len(actions)
        self.n_features = num_channel
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.num_channel = num_channel
        
        self.memory_size = 500
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))
        self.memory_counter = 0
        
        self.lstm_data = []
        self._build_lstm()
        
        self.model_eval = self._build_dqn()
        self.model_target = self._build_dqn()

        # ✅【修复1】必须用 legacy.Adam！！！
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.lr)
        # ✅【修复2】LSTM 也要单独一个优化器！
        self.optimizer_lstm = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

    def _build_dqn(self):
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(self.n_features,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.n_actions, activation='linear')
        ])
        return model

    def _build_lstm(self):
        self.model_lstm = models.Sequential()
        self.model_lstm.add(layers.LSTM(units=50, activation='relu', input_shape=(10, self.num_channel)))
        self.model_lstm.add(layers.Dense(units=self.num_channel, activation='sigmoid'))
        self.model_lstm.add(layers.Flatten())
        self.model_lstm.add(layers.Reshape((1, self.num_channel)))
        
    def channel_record(self, channel_state, access_act, action, su_index, num_su):
        channel_data_current = np.zeros(self.num_channel).astype(np.int32)
        for i in range(self.num_channel):
            channel_data_current[i] = channel_state[i]
        for i in range(num_su):
            if access_act[i] == 1 and i != su_index:
                channel_data_current[action[i]] = 1
        self.lstm_data.append(channel_data_current)

    def channel_prediction(self, step):
        if len(self.lstm_data) < 11:
            return np.zeros((1, self.num_channel))
            
        x_train = np.expand_dims(self.lstm_data[step - 11: step - 1], axis=0)
        y_train = np.array(self.lstm_data[step - 1: step])
        
        x_train_tensor = tf.convert_to_tensor(x_train, dtype=tf.float32)
        y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.float32)

        # ✅【修复3】LSTM 使用独立的 legacy 优化器
        with tf.GradientTape() as tape:
            y_pred = self.model_lstm(x_train_tensor, training=True)
            loss = tf.keras.losses.mean_squared_error(y_train_tensor, y_pred)
            loss = tf.reduce_mean(loss)
            
        gradients = tape.gradient(loss, self.model_lstm.trainable_variables)
        self.optimizer_lstm.apply_gradients(zip(gradients, self.model_lstm.trainable_variables))

        input_sequence = np.expand_dims(self.lstm_data[step - 10: step], axis=0)
        input_tensor = tf.convert_to_tensor(input_sequence, dtype=tf.float32)
        
        prediction = self.model_lstm(input_tensor, training=False)
        prediction = np.squeeze(prediction.numpy(), axis=0)
        
        return prediction

    def choose_action_lstm(self, observation, prediction):
        observation = np.array(observation, dtype=np.float32).reshape(1, -1)
        
        if np.random.uniform() < self.epsilon:
            q_values = self.model_eval(observation, training=False)[0].numpy()
            weighted_q_values = q_values * (1 - prediction[0])
            action = np.random.choice(np.where(weighted_q_values == np.max(weighted_q_values))[0])
        else:
            idle_probs = 1 - prediction[0] + 1e-5
            prob = idle_probs / np.sum(idle_probs)
            action = np.random.choice(self.actions, p=prob)
        return action

    def choose_action(self, observation):
        observation = np.array(observation, dtype=np.float32).reshape(1, -1)
        q_values = self.model_eval(observation, training=False).numpy()
        return np.argmax(q_values)

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self, s, a, r, s_):
        self.store_transition(s, a, r, s_)
        
        if self.memory_counter > self.memory_size:
            sample_size = 32
            indices = np.random.choice(self.memory_size, size=sample_size, replace=False)
            batch = self.memory[indices, :]
            
            states = batch[:, :self.n_features]
            actions = batch[:, self.n_features].astype(int)
            rewards = batch[:, self.n_features + 1]
            next_states = batch[:, -self.n_features:]
            
            states_tf = tf.convert_to_tensor(states, dtype=tf.float32)
            next_states_tf = tf.convert_to_tensor(next_states, dtype=tf.float32)
            rewards_tf = tf.convert_to_tensor(rewards, dtype=tf.float32)
            actions_tf = tf.convert_to_tensor(actions, dtype=tf.int32)

            with tf.GradientTape() as tape:
                q_eval = self.model_eval(states_tf)
                q_eval_selected = tf.reduce_sum(q_eval * tf.one_hot(actions_tf, self.n_actions), axis=1)
                
                q_next = self.model_target(next_states_tf)
                q_target = rewards_tf + self.gamma * tf.reduce_max(q_next, axis=1)
                
                loss = tf.keras.losses.mean_squared_error(q_target, q_eval_selected)
                loss = tf.reduce_mean(loss)

            gradients = tape.gradient(loss, self.model_eval.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model_eval.trainable_variables))

            # 定期更新 target 网络
            if self.memory_counter % 100 == 0:
                self.model_target.set_weights(self.model_eval.get_weights())