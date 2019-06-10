
from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
import data_provider
import gc
import poll
import user
import regression

class Autoencoder(object):
    """docstring for Autoencoder"""
    def __init__(self, model_name="model", num_epochs=1000, display_step=100, learning_rate=0.1, batch_size=100, 
        denoising=False, retrain_delay=10, graph_update_epochs=100, new_poll_weight=0.002,masking=0, num_layers=1, 
        num_hidden_1=155, num_hidden_2=128, continue_from_saved=False, content_collab_hybrid='collab'): 

        self.data_provider = data_provider.data_provider('../this_that_export_pretty.json')
        self.data_provider.parse()
        # Interactions are fed as binary but using decimal helps with adding 2 interactions together
        self.interaction_dict = {
            'skips': 16,
            'owns': 8,
            'tracks': 4,
            'comment': 2,
            'vote': 1
        }
        self.regression = regression.regression()
        self.num_engagements=len(self.interaction_dict)
        self.interactions_counter = 0 
        self.retrain_delay = retrain_delay # number of interactions needed before graph is trained more
        self.graph_update_epochs = graph_update_epochs # how long model is trained for on new interactions
        self.users = self.data_provider.users
        self.polls = self.data_provider.polls
        self.test_polls = self.data_provider.polls
        self.model_name = model_name
        self.num_epochs = num_epochs # initial training eppchs given training data
        self.display_step = display_step # display training loss every x epochs
        self.learning_rate = learning_rate 
        self.batch_size = batch_size
        self.denoising = denoising # whether to add noise to the input vectors - might help with accidental interactions
        self.new_poll_weight = new_poll_weight # How much weight new polls are given in the output layer (gives new polls some initial traction)
        self.masking = masking # TODO: Add masking to data to make synthetic users with less interactions and see if it helps
        self.num_layers = num_layers
        self.num_hidden_1 = num_hidden_1
        self.num_hidden_2 = num_hidden_2
        self.continue_from_saved = continue_from_saved
        self.train_proportion = 0.9
        self.train, self.test = [],[]
        self.test_users = []
        self.X = tf.placeholder("float", [None, None])
        self.Y = tf.placeholder("float", [None, None])
        self.weights2, self.weights1, self.biases2, self.biases1 = {},{},{},{}
        self.setup_graph()
        self.set_initial_training_and_test_data()

    def set_initial_training_and_test_data(self):
        indices = np.random.choice(range(len(self.users)), int(len(self.users)*self.train_proportion), replace=False)
        total_polls = len(self.polls)
        for idx,i in enumerate(self.users.keys()):
            poll_array = self.users[i].get_engagement_array(total_polls)
            if idx in indices:
                self.train.append(poll_array)
            else:
                self.test.append(poll_array)
                self.test_users.append(self.users[i])
        self.train = np.asarray(self.train)
        self.test = np.asarray(self.test)
    
    def setup_graph(self):
        
        # tf Graph input
        self.num_input = len(self.polls)*self.num_engagements
        tf.set_random_seed(1)
        self.weights2 = {
            'encoder_h1': tf.Variable(tf.random_normal([self.num_input, self.num_hidden_1])),
            'encoder_h2': tf.Variable(tf.random_normal([self.num_hidden_1, self.num_hidden_2])),
            'decoder_h1': tf.Variable(tf.random_normal([self.num_hidden_2, self.num_hidden_1])),
            'decoder_h2': tf.Variable(tf.random_normal([self.num_hidden_1, self.num_input])),
        }
        self.biases2 = {
            'encoder_b1': tf.Variable(tf.random_normal([self.num_hidden_1])),
            'encoder_b2': tf.Variable(tf.random_normal([self.num_hidden_2])),
            'decoder_b1': tf.Variable(tf.random_normal([self.num_hidden_1])),
            'decoder_b2': tf.Variable(tf.random_normal([self.num_input])),
        }
        self.weights1 = {
            'encoder_h1': tf.Variable(tf.random_normal([self.num_input, self.num_hidden_1])),
            'decoder_h1': tf.Variable(tf.random_normal([self.num_hidden_1, self.num_input])),
        }
        self.biases1 = {
            'encoder_b1': tf.Variable(tf.random_normal([self.num_hidden_1])),
            'decoder_b1': tf.Variable(tf.random_normal([self.num_input])),
        }
        if self.num_layers == 2:
            self.encoder_op = self.encoder2(self.X)
            self.decoder_op = self.decoder2(self.encoder_op)
        else:
            self.encoder_op = self.encoder1(self.X)
            self.decoder_op = self.decoder1(self.encoder_op)

        # Prediction
        self.y_pred = self.decoder_op
        # Targets (Labels) are the input data.
        self.y_true = self.Y
        # Define loss and optimizer, minimize the squared error
        # self.loss = tf.reduce_mean(tf.pow(self.y_true[:self.true_input*self.engagement_options] - self.y_pred[:self.num_input*self.engagement_options], 2))
        self.loss = self._loss(self.y_true, self.y_pred)
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

        # Initialize the variables (i.e. assign their default value)
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

    def _loss(self,y_true, y_pred):
        coef = tf.Variable(self.regression.reg.coef_, dtype=tf.float32)
        intercept = tf.Variable(self.regression.reg.intercept_,dtype=tf.float32)
        y_true = tf.math.add(tf.math.multiply(coef, tf.reshape(y_true,[-1,5])), intercept)
        y_pred = tf.math.add(tf.math.multiply(coef, tf.reshape(y_pred,[-1,5])), intercept)
        return tf.reduce_mean(tf.pow(self.y_true - self.y_pred, 2))


    def encoder1(self,x):
        # Encoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights1['encoder_h1']),
                                       self.biases1['encoder_b1']))
        return layer_1

    def decoder1(self,x):
        # Decoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights1['decoder_h1']),
                                       self.biases1['decoder_b1']))
        return layer_1

    def encoder2(self,x):
        # Encoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights2['encoder_h1']),
                                       self.biases2['encoder_b1']))
        # Encoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights2['encoder_h2']),
                                       self.biases2['encoder_b2']))
        return layer_2

    def decoder2(self,x):
        # Decoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights2['decoder_h1']),
                                       self.biases2['decoder_b1']))
        # Decoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights2['decoder_h2']),
                                       self.biases2['decoder_b2']))
        return layer_2

    def initial_trainer(self, save=True):
        gc.collect()
        with tf.Session() as sess:

            # Run the initializer
            sess.run(tf.global_variables_initializer())
            # Training
            train_loss = []
            test_loss = []
            t0 = time.time()
            for i in range(1, self.num_epochs):
                # Prepare Data
                batch_x = self.train[np.random.choice(self.train.shape[0], self.batch_size, replace=True), :]
                batch_y = batch_x
                if self.denoising:
                    noise = np.random.normal(0.05, 0.1, batch_x.shape)
                    batch_x = np.add(batch_x,noise)
                if self.masking > 0:
                    pass
                batch_test = self.test

                # Run optimization op (backprop) and cost op (to get loss value)
                _, l = sess.run([self.optimizer, self.loss], feed_dict={self.X: batch_x, self.Y: batch_y})
                train_loss.append(l)
                _, loss_test = sess.run([self.decoder_op, self.loss], feed_dict={self.X: batch_test, self.Y: batch_test})
                test_loss.append(loss_test)
                # Display logs per step
                if i % self.display_step == 0 or i == 1:
                    print('Step %i: Minibatch Loss: %f Test Loss %f' % (i, l, loss_test))
            if save:
                self.saver.save(sess, 'refactored/model.ckpt')
            t1 = time.time()
            total_time = t1-t0
            overfitting = train_loss[-1] - test_loss[-1]
            sess.close()
            return test_loss[-1], total_time, overfitting 

    def trainer(self, num_epochs=100, num_recent_users=30):
        train_data = self.get_most_recently_active_user_vectors(num_recent_users)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.saver.restore(sess, 'refactored/model.ckpt')
            for i in range(1, num_epochs):
                batch_x = train_data[np.random.choice(train_data.shape[0], self.batch_size, replace=True), :]
                batch_y = batch_x
                if self.denoising:
                    noise = np.random.normal(0.05, 0.1, batch_x.shape)
                    batch_x = np.add(batch_x,noise)
                if self.masking > 0:
                    pass
                # Run optimization op (backprop) and cost op (to get loss value)
                _, l = sess.run([self.optimizer, self.loss], feed_dict={self.X: batch_x, self.Y: batch_y})
                # train_loss.append(l)

            self.saver.save(sess, 'refactored/model.ckpt')

    def get_most_recently_active_user_vectors(self,num):
        lastActives = []
        vecs = []
        for i in self.users.keys():
            lastActives.append(self.users[i].lastActive)
        for i in np.argsort(-np.asarray(lastActives))[:num]:
            vecs.append(self.users[self.users.keys()[i]].get_engagement_array(len(self.polls)))
        return np.asarray(vecs)

    def predict(self, userid, num_recs=10):
        # TODO:
        # - Add variation in engagement values
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            u_vector_length = len(self.polls)
            user_vector = self.users[userid].get_engagement_array(u_vector_length).reshape(-1,u_vector_length*self.num_engagements)
            self.saver.restore(sess, 'refactored/model.ckpt')
            res = sess.run([self.decoder_op], feed_dict={self.X: user_vector})
            poll_expectation = np.sum((np.asarray(res).reshape(-1,self.num_engagements)), axis=1) # add specific engagement sum here
            true_expectation = np.sum((np.asarray(user_vector).reshape(-1,5)), axis=1)
            timed_polls = range(len(self.polls))
            scores = []
            true_scores = []
            timed_scores = []
            for i in np.argsort(-poll_expectation):
                # Never recommend a poll that a user has already interacted with
                if self.polls[i].pollid not in self.users[userid].polls.keys():
                    scores.append([self.poll_score(poll_expectation[i], self.polls[i]), self.polls[i].pollid])
                    true_scores.append([self.poll_score(true_expectation[i], self.polls[i]), self.polls[i].pollid])
                    timed_scores.append([self.poll_score(timed_polls[i], self.polls[i]), self.polls[i].pollid])
            scores = sorted(scores, key=lambda x: x[0], reverse=True)
            true_scores = sorted(true_scores, key=lambda x: x[0], reverse=True)
            timed_scores = sorted(timed_scores, key=lambda x: x[0], reverse=True)
            return [self.order_difference(true_scores, scores),self.order_difference(true_scores, timed_scores), 
            self.precision_at_50(true_scores, scores), self.precision_at_50(true_scores, timed_scores)]
            # print(scores[:num_recs])

    def poll_score(self, poll_expectation, poll):
        time_ago = time.gmtime(time.time() - poll.timestamp).tm_hour
        return (poll_expectation / (time_ago+0.001)**1.2) + poll.boost

    def order_difference(self,truelist, predictedlist):
        score = 0
        truelist = [x[1] for x in truelist]
        for idx,i in enumerate(predictedlist):
            score += np.absolute(truelist.index(i[1]) - idx)**2
        return np.sqrt(score/len(predictedlist))

    def precision_at_100(self, truelist, predictedlist):
        p = 0 
        truelist = [x[1] for x in truelist[:100]]
        for i in predictedlist[:100]:
            if i[1] in truelist:
                p+=1
        return p/100

    def new_poll(self,userid,pollid,timestamp,title,this,that):
        new_poll = poll.Poll(userid,pollid, timestamp, title, this, that)
        self.polls.append(new_poll)
        self.update_graph()
        self.add_interaction([userid],[[pollid]], [[[self.interaction_dict['owns']]]])
        
    def new_user(self, userid, lastActive):
        u = user.User(userid, lastActive)
        if userid not in self.users.keys():
            self.users[userid] = u

    def update_graph(self):

        # tf Graph input
        self.num_input = len(self.polls)*self.num_engagements

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            var = sess.run(self.weights1['encoder_h1'])
            new = np.zeros(shape=(self.num_engagements,self.num_hidden_1)).astype('float32')
            var = np.append(var, new, axis=0)
            self.weights1['encoder_h1'] = tf.Variable(var)


            var = sess.run(self.weights1['decoder_h1'])
            new = np.full(shape=(self.num_engagements,self.num_hidden_1), fill_value=self.new_poll_weight).astype('float32')
            var = np.append(np.swapaxes(var,0,1), new, axis=0)
            self.weights1['decoder_h1'] = tf.Variable(np.swapaxes(var,0,1))


            # var = sess.run(self.biases1['encoder_b1'])
            # new = np.random.normal(1,size=(4,)).astype('float32')
            # var = np.append(var, new, axis=0)
            # self.biases1['encoder_b1'] = tf.Variable(var)

            var = sess.run(self.biases1['decoder_b1'])
            new = np.random.normal(1,size=(self.num_engagements,)).astype('float32')
            var = np.append(var, new, axis=0)
            self.biases1['decoder_b1'] = tf.Variable(var)

            if self.num_layers == 2:
                self.encoder_op = self.encoder2(self.X)
                self.decoder_op = self.decoder2(self.encoder_op)
            else:
                self.encoder_op = self.encoder1(self.X)
                self.decoder_op = self.decoder1(self.encoder_op)
            self.saver.save(sess, 'refactored/model.ckpt')
        # Prediction
        self.y_pred = self.decoder_op
        # Targets (Labels) are the input data.
        self.y_true = self.Y
        # Define loss and optimizer, minimize the squared error
        # self.loss = tf.reduce_mean(tf.pow(self.y_true[:self.true_input*self.engagement_options] - self.y_pred[:self.num_input*self.engagement_options], 2))
        self.loss = tf.reduce_mean(tf.pow(self.y_true - self.y_pred, 2))
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

        # Initialize the variables (i.e. assign their default value)
        self.init = tf.global_variables_initializer()
        # self.saver = tf.train.Saver()

    # Batch add interactions in the form:
    # (userids:[1,2,3], pollids:[[32,2,4],[334,2],[2]], interactions:[[8,2],[4],[2]])
    def add_interaction(self, userids, pollids, interactions):
        for idx,i in enumerate(userids):
            for jdx,j in enumerate(pollids[idx]):
                for k in interactions[idx][jdx]:
                    self.users[i].add_interaction(j, k)
                    self.interactions_counter += 1
                    if self.interactions_counter % self.retrain_delay == 0:
                        self.interactions_counter = 0 
                        self.trainer(self.graph_update_epochs)


# autoencoder_results_rmse = []
# baseline_results_rmse = []
# autoencoder_results_prec = []
# baseline_results_prec = []
# for i in range(20,500)[::20]:

#     a = Autoencoder(num_epochs=i)
#     a.initial_trainer(save=True)
#     mae_auto, mae_base = [],[]
#     prec_auto, prec_base = [],[]
#     for idx,i in enumerate(a.test_users):
#         res = a.predict(i.userid)
#         mae_auto.append(res[0])
#         mae_base.append(res[1])
#         prec_auto.append(res[2])
#         prec_base.append(res[3])
#     autoencoder_results_rmse.append(np.mean(mae_auto))
#     baseline_results_rmse.append(np.mean(mae_base))
#     autoencoder_results_prec.append(np.mean(prec_auto))
#     baseline_results_prec.append(np.mean(prec_base))
#     print("autoencoder RMSE: %f \nbaseline RMSE:" % np.mean(mae_auto), np.mean(mae_base))
#     print("autoencoder precision_at_50: %f \nbaseline precision_at_50:" % np.mean(prec_auto), np.mean(prec_base))

# print(autoencoder_results_rmse)
# print(baseline_results_rmse)
# print(autoencoder_results_prec)
# print(baseline_results_prec)

# plt.figure(0)
# plt.plot(autoencoder_results_rmse, label='Autoencoder')
# plt.plot(baseline_results_rmse, label='Baseline')
# plt.xlabel('Training epochs')
# plt.ylabel('Prediction RMSE')
# plt.title('RMSE of poll prediction on test users \ncompared to ideal poll prediction')
# plt.savefig('rmse_prediction.pdf')

# plt.figure(1)
# plt.plot(autoencoder_results_prec, label='Autoencoder')
# plt.plot(baseline_results_prec, label='Baseline')
# plt.xlabel('Training epochs')
# plt.ylabel('Prediction Precision')
# plt.title('Precision_at_50 of poll prediction on test users \ncompared to ideal poll prediction')
# plt.savefig('prec_prediction.pdf')


total_interactions = 0
a = Autoencoder()
a.initial_trainer()
# for i in a.users.keys():
#     total_interactions += a.users[i].num_interactions()

# print(float(total_interactions/len(a.users)))
# print(len(a.data_provider.polls))
# print(len(a.users))








