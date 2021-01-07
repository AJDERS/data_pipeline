import configparser
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.layers import Dropout, Dense, LSTMCell, RNN, Masking, Flatten, Concatenate
from tensorflow.keras import Model



class RNNStepPredictor(Model):
    def __init__(self, sliding_windows, config_path):
        super().__init__()
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.sliding_windows = sliding_windows # Should be instance of `SlidingWindow`
        self.activation = self.config['RNN'].get('Activation')
        self.lstm_units = self.config['RNN'].getint('LSTMUnits')
        self.fc_units = self.config['RNN'].getint('FCUnits')
        self.batch_size = self.config['PREPROCESS_TRAIN'].getint('BatchSize')
        num_scatter_tmp = self.config['DATA'].get('NumScatterTrain')
        self.num_scatterer =  [int(y) for y in num_scatter_tmp.split(',')][0]
        self.prior = self.config['RNN'].getint('Prior')
        self.future = self.config['RNN'].getint('Future')
        self.droprate = self.config['RNN'].getfloat('DropOutRate')
        self.window_length = self.prior + self.future
        self.cost_weight = self.config['RNN'].getfloat('CostWeight')
        self._make_lstm()

    def _make_lstm(self):
        self.lstm_cell = LSTMCell(self.lstm_units, dtype=tf.float64)
        self.lstm_rnn = RNN(self.lstm_cell, return_state=True)

    def _activation(self):
        if self.activation == 'tanh':
            act = tf.nn.tanh
        elif self.activation  == 'relu':
            act = tf.nn.relu
        elif self.activation  == 'relu6':
            act = tf.nn.relu6
        elif self.activation  == 'crelu':
            act = tf.nn.crelu
        elif self.activation  == 'elu':
            act = tf.nn.elu
        elif self.activation  == 'softsign':
            act = tf.nn.softsign
        elif self.activation  == 'softplus':
            act = tf.nn.softplus
        elif self.activation  == 'sigmoid':
            act = tf.sigmoid
        else:
            act = None
        self.activation = act

    def _get_extensions(self, batch, time):
        extensions = []
        for i in range(self.batch_size):
            batch_list = []
            time_features = [x for x in batch[i] if x[2]==time]
            for s in range(self.num_scatterer):
                scat_sorted_time_features = []
                for feature in time_features:
                    if feature[1] == s:
                        scat_sorted_time_features.append(feature)
                batch_list.append([self._redo(x[3][2])[-1,:] for x in scat_sorted_time_features])
            extensions.append(batch_list)
        return extensions

    def _warmup(self, start_features):
        """
        For each coordinate the LSTM predicts the next step.
        This method is for warm-up i.e. the first `window_size`-length
        set of frames are passed to the LSTM, and it predicts the steps from 1
        to `window_size`+1. All steps and the final state are returned. 
        """
        # start_features.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        # # predictions.shape => (batch, features)
        mask = Masking(mask_value=-1.0, dtype=np.float64)
        start_features = mask(start_features)
        def lstm_prediction(data, state):
            x, state = self.lstm_cell(data, states=state)
            x = Dropout(rate=self.droprate, dtype='float64')(x)
            prediction = Dense(units=2, activation=self.activation)(x)
            return prediction, x, state

        predictions = []
        lstm_outputs = []
        state = self.lstm_cell.get_initial_state(
            inputs=start_features[:,0],
            batch_size=self.batch_size,
            dtype=np.float64
        )
        for t in range(self.window_length):
            prediction, x, state = lstm_prediction(start_features[:,t], state=state)
            predictions.append(prediction)
            lstm_outputs.append(x)
        predictions = tf.stack(predictions)
        predictions = tf.transpose(predictions, [1, 0, 2])
        lstm_outputs = tf.stack(lstm_outputs)
        lstm_outputs = tf.transpose(lstm_outputs, [1, 0, 2])
        return predictions, lstm_outputs, state

    def _redo(self, data):
        return np.transpose(np.squeeze(data), (1,0))

    def _get_start_features(self, batch):
        start_features = np.zeros((self.batch_size, self.num_scatterer, self.window_length, 2))
        for i in range(self.batch_size):
            time_zero_features = [x for x in batch[i] if x[2]==0]
            for s in range(self.num_scatterer):
                start_features[i, s, :, :] = self._redo(time_zero_features[s][3][2])
        return start_features

    def _fully_connected(self, tensor, order):
        assert order in ['first', 'second', 'after_concat']
        if order == 'first':
            tensor = np.reshape(tensor, (1,2))
            tensor = Dense(self.fc_units, activation=self.activation)(tensor)
            tensor = Dropout(self.droprate,dtype='float64')(tensor)
        elif order == 'second':
            tensor = Dense(self.fc_units, activation=self.activation)(tensor)
            tensor = Dropout(self.droprate, dtype='float64')(tensor)
        else:
            tensor = Dense(self.fc_units, activation=self.activation)(tensor)
            tensor = Dropout(self.droprate, dtype='float64')(tensor)
        return tensor

    def _calc_true_association(self, candidate, time, label_data):
        label = label_data[time][3][:,-1, 0]
        true_cost = np.linalg.norm(label-candidate)
        if true_cost == 0.0:
            binary_label = np.array([1.0, 0.0])
        else:
            binary_label = np.array([0.0, 1.0])
        return label, true_cost, binary_label

    def _dist_candidate_to_lstm(self, candidate, lstm_predictions, batch_index):
        if len(lstm_predictions.shape) != 2:
            predicted_coord = lstm_predictions[batch_index, -1, :]
        else:
            predicted_coord = lstm_predictions[batch_index, :]
        return np.linalg.norm(candidate-predicted_coord)
        

    def _loss(
        self,
        association_cost,
        true_cost,
        association_prop,
        binary_label):
        cost_loss = tf.squeeze(self.cost_weight * (association_cost - true_cost) ** 2)
        prob_loss = tf.math.reduce_sum(
            tf.math.multiply(binary_label, tf.math.log(association_prop))
        )
        return cost_loss - prob_loss


    def call(self, inputs):
        """
        Given a batch consisting of a tensor of shape:
        (batch_size, num_scatterer, coords, time, channel)

        a list of tuples containing:
        (index, scatterer_index, time, (previous_candidate, candidate, window))
        is produced. For each scatterer the first `window_length` coordinates 
        are found, via `_get_start_features`. For each of these initial
        features their possible extensions, i.e. the possible 
        `window_length + 1` coordinates are found via `_get_extensions`.

        The initial features are passed to a LSTM-layer in `_warmup` which
        sets initial states, does warm up of LSTM with initial features and
        predicts next extension. The predictions are results of a FC-layer with
        2 units, corresponding to the two coordinates. The LSTM output, 
        predictions and states are returned.

        The LSTM outputs are flattened and concatenated with the output of 
        a FC-layer applied to each extension. This concatenate tensor is fed
        to another FC-layer, which output is used for evaluating the
        association cost and positive/negative-association probability.

        From the LSTM prediction the candidate which is closest to the 
        prediction is found. The candidate is stored in `predictions` and its
        corresponding association costs and probabilities are stored in
        `association_costs` and `association_probabilities` respectively.

        This concludes the flow for the first time step. For the next time step
        the prediction is feed to the LSTM-layer which produces a new
        prediction. This prediction and the new possible extensions are found
        are used as input to the next time step. The flow is repeated
        `self.window_length` times.
        """
        # Start
        # inputs: (batch_size, num_scatterer, coords, time, channel)
        
        # Make sliding windows.
        window_data = []
        for i in range(inputs.shape[0]):
            x = self.sliding_windows.run(tf.expand_dims(inputs[i], axis=0))
            window_data.append(x)

        # window_data: [(index, scatterer_index, time, (previous_candidate, candidate, window))]

        # Preallocate output tensors.
        # (batch_size, future, coords)
        output = np.zeros((3, self.batch_size, self.num_scatterer, self.window_length, 2))
        predictions = np.zeros((self.batch_size, self.num_scatterer, self.window_length, 2))
        association_probabilities = np.zeros((self.batch_size, self.num_scatterer, self.window_length, 2))
        association_costs = np.zeros((self.batch_size, self.num_scatterer, self.window_length, 2))

        # Get start features, i.e. first window from each scatter
        start_features = self._get_start_features(window_data)
        # start_features: (batch_size, window_size, coords)

        for s in range(self.num_scatterer):
            # Initialize LSTM cell, this is time 0.
            lstm_predictions, lstm_outputs, state = self._warmup(start_features[:,s])
            # lstm_predictions: (batch_size, window_size, coords)
            # states: list[(batch_size, units), (batch_size, units)]
            # lstm_outputs: (batch_size, window_size, lstm_units)

            time = 1
            # Get extensions, i.e. coords for each candidate at a specific time
            extensions = self._get_extensions(window_data, time)
            # extensions: list[list[(number_of_candidates, coords)]]
            # Outer list is len == batch_size, inner list is len == num_scatterer


            # Note time index starts at 0, while extensions start from time 1, hence `-1` or `+1`
            flat_lstm_outputs = Flatten()(lstm_outputs)
            while time < self.window_length + 1:
                for batch_index, extension in enumerate(extensions):
                    cost_values = []
                    distance_values = []
                    prob_values = []
                    for candidate in extension[s]:
                        # Get distances from candidate to LSTM predictions
                        dist_to_predicted_coord = self._dist_candidate_to_lstm(
                            candidate,
                            lstm_predictions,
                            batch_index
                        )
                        distance_values.append(dist_to_predicted_coord)

                        # Predict association cost and probability
                        candidate_tensor = self._fully_connected(candidate, 'first')
                        candidate_tensor = self._fully_connected(candidate_tensor, 'second')
                        flat_candidate_tensor = Flatten()(candidate_tensor)
                        concat_tensor = Concatenate()(
                            [
                                tf.reshape(flat_lstm_outputs[batch_index], (1, self.window_length * self.lstm_units)),
                                flat_candidate_tensor
                            ]
                        )
                        evaluation_tensor = self._fully_connected(concat_tensor, 'after_concat')
                        
                        association_prop = Dense(2, activation='softmax')(evaluation_tensor)
                        prob_values.append(association_prop)

                        association_cost = Dense(1, activation=None)(evaluation_tensor)
                        cost_values.append(association_cost)

                    # Get LSTM predicted candidate.
                    distance_values = np.array(distance_values)
                    min_dist_index = np.where(distance_values==distance_values.min())[0][0]

                    # Store prediction, cost, prob
                    # Note time index starts at 0, while extensions start from time 1, hence `-1`
                    predictions[batch_index, s, time-1, :] = extension[s][min_dist_index]
                    association_costs[batch_index, s, time-1, :] = cost_values[min_dist_index]
                    association_probabilities[batch_index, s, time-1, :] = prob_values[min_dist_index]

                # Setup for next time step
                time += 1
                if not time == self.window_length+1:
                    extensions = self._get_extensions(window_data, time)
                    lstm_outputs, state = self.lstm_cell(predictions[:, s, time-1], states=state)
                    x = Dropout(rate=self.droprate, dtype='float64')(lstm_outputs)
                    lstm_predictions = Dense(units=2, activation=self.activation)(x)
                

            # Output single value to fit with custom loss function API.
            output[0,:] = predictions
            output[1,:] = association_costs
            output[2,:] = association_probabilities
        return output


# FIX: MASKING AND CAPPING

