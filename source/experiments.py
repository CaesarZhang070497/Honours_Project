
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
        denoising=False, retrain_delay=5, graph_update_epochs=1000, new_poll_weight=0.002,masking=0, num_layers=1, num_hidden_1=155, num_hidden_2=128, 
        continue_from_saved=False, content_collab_hybrid='collab', time_decay=1): 

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
        self.users = self.data_provider.users#[:50]
        self.polls = self.data_provider.polls#[:50]
        self.test_polls = self.data_provider.polls#[500:]
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
        self.train_proportion = 0.85
        self.train, self.test = [],[]
        self.test_users = []
        self.time_decay = time_decay
        self.X = tf.placeholder("float", [None, None])
        self.Y = tf.placeholder("float", [None, None])
        self.saver = None
        self.weights2, self.weights1, self.biases2, self.biases1 = {},{},{},{}
        self.setup_graph()
        self.set_initial_training_and_test_data()

    def set_initial_training_and_test_data(self):
        indices = np.random.choice(range(len(self.users)), int(len(self.users)*self.train_proportion), replace=False)
        total_polls = len(self.polls)
        for idx,i in enumerate(self.users.keys()):
            poll_array = self.users[i].get_engagement_array(total_polls)
            # print(self.users[i].num_interactions())
            if idx in indices:
                self.train.append(poll_array)
            else:
                if self.users[i].num_interactions() > 40 and self.users[i].num_interactions() < 100:
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
        # coef = tf.Variable([5, 3, 4, 2, 1], dtype=tf.float32)
        # intercept = tf.Variable(self.regression.reg.intercept_,dtype=tf.float32)
        # y_true = tf.math.add(tf.math.multiply(coef, tf.reshape(y_true,[-1,5])), intercept)
        # y_true = tf.math.multiply(coef, tf.reshape(y_true,[-1,5]))
        # y_pred = tf.math.add(tf.math.multiply(coef, tf.reshape(y_pred,[-1,5])), intercept)
        # y_pred = tf.math.multiply(coef, tf.reshape(y_pred,[-1,5]))
        return tf.reduce_mean(tf.pow(self.y_true - self.y_pred, 2))


    def encoder1(self,x):
        # Encoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights1['encoder_h1']),
                                       self.biases1['encoder_b1']))
        # layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, self.weights1['encoder_h1']),
        #                                self.biases1['encoder_b1']))
        return layer_1

    def decoder1(self,x):
        # Decoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights1['decoder_h1']),
                                       self.biases1['decoder_b1']))
        # layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, self.weights1['decoder_h1']),
        #                                self.biases1['decoder_b1']))
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
            # self.saver.restore(sess, 'refactored/model.ckpt')

            # Training
            train_loss = []
            test_loss = []
            t0 = time.time()
            test_accuracy = []
            test_precision = []
            for i in range(0,self.num_epochs):
                # Prepare Data
                batch_x = self.train[np.random.choice(self.train.shape[0], self.batch_size, replace=True), :]
                batch_y = np.copy(batch_x)
                if self.denoising:
                    # noise = np.random.sample(batch_x.shape) * 0.01
                    noise = np.random.choice(2, batch_x.shape, p=[0.98,0.02])
                    z = np.where(batch_x < 1)
                    batch_x[z] = noise[z]
                    # batch_x = np.add(batch_x[z],noise[z])
                z = np.where(batch_x > 0)
                if self.masking > 0:
                    noise = np.random.choice(2, batch_x.shape, p=[(self.masking),(1-self.masking)])
                    
                    batch_x[z] = noise[z]
                # batch_test = self.test
                # batch_y_test = batch_test
                # for jdx,j in enumerate(batch_test):
                #     indices = np.argwhere(j>0)
                #     if len(indices) > 0:
                #         mask = np.random.choice(indices.flatten(), int(len(indices*0.2)))
                #         for k in mask:
                #             batch_test[jdx][k] = 0 
                # Run optimization op (backprop) and cost op (to get loss value)

                _, l = sess.run([self.optimizer, self.loss], feed_dict={self.X: batch_x, self.Y: batch_y})
                train_loss.append(l)
                # _, loss_test = sess.run([self.decoder_op, self.loss], feed_dict={self.X: batch_test, self.Y: batch_test})
                # test_loss.append(loss_test)
                # Display logs per step
                if save and (i == 0 or i % self.display_step == 0):
                    self.saver.save(sess, 'refactored/model.ckpt')
                if i % self.display_step == 0 or i == 1:
                    accs = []
                    precs = []
                    baseline_precs = []
                    devs = []
                    for u in self.test_users:
                        a, b, c, d, e = self.predict(u.userid)
                        # accs.append(self.predict(u.userid,10)[0])
                        # precs.append(self.predict(u.userid,10)[2])
                        accs.append(a)
                        precs.append(c)
                        baseline_precs.append(d)
                        devs.append(e)
                    test_accuracy.append(np.mean(np.asarray(accs)))
                    test_precision.append(np.mean(np.asarray(precs)))
                    print('Step %i: Minibatch Loss: %f Test Loss %f Test precision %f baseline: %f' % (i, l, test_accuracy[-1], test_precision[-1], baseline_precs[-1]))
                    # print(test_accuracy[-1])
                
                
            t1 = time.time()
            total_time = t1-t0
            return test_precision
            # overfitting = train_loss[-1] - test_loss[-1]
            sess.close()
            # return test_loss[-1], total_time, overfitting 

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
        lastActives = [u'ssauRCxp4gMDGYm9w4jFwxWabvG2', u'HZ6ZcUVHeXYmZIxwwMrQm7WNqfH3', u'fChiYAtps9hy2pXgfRe8APAFgeT2', u'kEZfgYC7FRRLq78sa0C1oDpuPZ13', u'rv4PMYMToXN2oIoYoYcXs8k8kUC3', u'Vrv2bxkdZ4S19ECBQ7pWhrLkdRC2', u'o8BLoR68Ylgnk88rKlKd8uNhFN32', u'xAVTUQHIRNdXpryNn2yUfALbk892', u'UX0Os30KbXRAOTYSKs2BLGaz61k2', u'y0RqzcIEycRGsEJog7z8SbbvNb13', u'REnpwB7yGDe1whBlwqfAfuIGyWS2', u'SJYZdf8Vk4PNoWA7gm3Cwe62IwC3', u'bUgIR9tAJGgmlxCkfVBgbM41V7B2', u'dV7q7Blcpjd4IBRUhCIr1FbBAlA3', u'ut2a2ZN37zRNgePREeNde2a79bE3', u'G5fXwWOC01ccMVvEiSTZ9BsJOH13', u'L7S2I2u5psfTDqEA30aDSWE7h2g2', u'UknrMnapbFcaT6DMs2OSaJjrKZ72', u'aDPjMUajnDQIvNdGpGh9gmFRA2k2', u'A3cxNwOMssPsKw0RmcS3QqYGLoj1', u'krO9dUgbEJeavIjlxTUSgd9Hef62', u'7dBSwgcgirgaBKXD73Dh075zKel1', u'eaxElYr0iKfWNtrVmcren7OEaDw1', u'F4ofvTitUzfBHJbbgPgjQH7x1ko2', u't7ov5qEFNPOPgpkJiMjHgvh3IyX2', u'qzM8Kp884JchdSmc9zXkuGWkwFm1', u'bavjkXVtDIbPTFNhKdXQWNxp7s43', u'MV5fbiGluYWaR5KJnBPXwyByc9l1', u'AWRZpNVejxMzmNXcuMBBEWg36FJ2', u'v2Y1q8B9wrO9SknQzTvKi1aTUXc2', u'3sAqU9MhkkdFd8zxNlXCdmJd4Lu1', u'5PLvJeJttUasuJFM5vXAkh8sa5G2', u'NZp7LGUvKLZPoEvvLSqnAxYFG7t1', u'SodaZQhSL9UjxJ4vRDeimlG8xAK2', u'YnkBuWeuZlZk6tvMtCfDZ5ac2RJ3', u'cR5YLdfFBEYNgyYnp7JfFmlpGM92', u'45f5hYDoeNY9MSSONZ0tlHt8pbr2', u'RRV1VIDtkET6ovdAtUHPZQVs1RP2', u'gv3YXCU3nebRft1DDXOkeh7g09g1', u'wr2Tf0DrY9duPHZEz687BoG84Bi1', u'mOSXPOgyxUTjUCA3x0kp59GE2Sm2', u'ZjuFGLqUuxdXQPIkSg6smIsYrMI3', u'z6kKIG0L7ccNP6cg9dEEAnsRFgk1', u'hoLXtDQpmBWqTI0liRCoWfHx4tF2', u'cjasuQmHhBbzaxih4LYpOB9WXRt2', u'DNR8idL2NiPv9mP4JjwmYekQyBF2', u'uCIUbhJ4UFheU7okKjlHo3Y1Hma2', u'L0N9vgBLAlTKOk24pQ2BCoVMmr12', u'ue5fRVG1seQvpukudDouMzrL8Gp2', u'6aoC81SXC2YsLsBoT1aLr2BvK993', u'Y4tgpPHQdTcLQ1NLOAal1K90ftx1', u'3W4pzU6zUVapSthzkG498cC2hbM2', u'PkEnZ9IJuMcK0k7hjr5aZkHZCrq2', u'Cv2fTDo9haTgrpYG2fkKFWSYWXA2', u'isFsisiQFzY8UXO8EgK6Bbzwp4n2', u'hffP2jBG2bPnrmNhAXFOEwQEpjD2', u'xXxm8kENK9elP8HdkrBomWJjSnH2', u'CnhrSwhP8GXdZazlrSrerFAURxz1', u'NowgqkLsAwcJiCWbUHw41osoXUE3', u'Bbd8BqEtqyTlkZWdJZ0Yhv6hApp1', u'vFvM2fv7OFhoH9iH4QRBHnITpBX2', u'nSoDWSANS5VZGtmmONCVS02nT8n1', u'qtHQrAdTJyO1Mw0GP6lygfj0e0j1', u'uGeX3IbQ1ifSkpePtuHGXx2ZwW83', u'WmlVl2vAwTYlNr387rhfQDTY2S63', u'thGGbLDwftgvKZfwQz0RuCr0ufU2', u'58tnxZ2MdRfEOxpWTsTVKyr9ryw1', u'tzIsQULp22W1Hsmv2WED6TIdFMJ3', u'SDZXp0np8fSPWvi4GQ4RKsrH2dL2', u'fc4IbdViIXPvqXTpt6LxWctbXwn2', u'TIvCzYKDLtaxLUAF1hzs44HnJ5Z2', u'UFTzdO1OBDV1WkNtaIEf5umcz4E2', u'xC46hES2oxfOIA7eSbYH0z4zThe2', u'ZEzJAjZ2miPeBSOPkDLtmL0sTCn2', u'Pktw7H1sWDPzUzQnAC4Ui9zQIm63', u'Mew5o6fgpFU6w7AWwEHLoHcTViA3', u'J0MM8WIQvtXpu7b0EFKiQYmuBWL2', u'xC46hES2oxfOIA7eSbYH0z4zThe2', u'PkEnZ9IJuMcK0k7hjr5aZkHZCrq2', u'v2Y1q8B9wrO9SknQzTvKi1aTUXc2', u'cR5YLdfFBEYNgyYnp7JfFmlpGM92', u'5PLvJeJttUasuJFM5vXAkh8sa5G2', u'G5fXwWOC01ccMVvEiSTZ9BsJOH13', u'hffP2jBG2bPnrmNhAXFOEwQEpjD2', u'isFsisiQFzY8UXO8EgK6Bbzwp4n2', u'qzM8Kp884JchdSmc9zXkuGWkwFm1', u'CnhrSwhP8GXdZazlrSrerFAURxz1', u'Mew5o6fgpFU6w7AWwEHLoHcTViA3', u'z6kKIG0L7ccNP6cg9dEEAnsRFgk1', u'DNR8idL2NiPv9mP4JjwmYekQyBF2', u'uGeX3IbQ1ifSkpePtuHGXx2ZwW83', u'HZ6ZcUVHeXYmZIxwwMrQm7WNqfH3', u'L0N9vgBLAlTKOk24pQ2BCoVMmr12']
        vecs = []
        # for i in self.users.keys():
        #     lastActives.append(self.users[i].lastActive)
        for i in lastActives[:num]:
            vecs.append(self.users[i].get_engagement_array(len(self.polls)))
        return np.asarray(vecs)

    def predict(self, userid, num_recs=50):
        # TODO:
        # - Add variation in engagement values
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            u_vector_length = len(self.polls)
            user_vector = self.users[userid].get_engagement_array(u_vector_length).reshape(-1,u_vector_length*self.num_engagements)
            truthy = np.copy(user_vector).reshape(-1,5)
            mask = []
            # for jdx,j in enumerate(user_vector):
            indices = []
            for idx,i in enumerate(user_vector.reshape(-1,5)):
                if np.sum(i) > 0 and i[0] < 1:
                    indices.append(idx)
            # indices = np.argwhere(user_vector[0] > 0)
            mask = []
            if len(indices) > 0:
                mask = np.random.choice(indices, int(len(indices)*0.2))
                for k in mask:
                    user_vector[0][k] = 0        
            self.saver.restore(sess, 'refactored/model.ckpt')
            res = sess.run([self.decoder_op], feed_dict={self.X: user_vector})
            res = np.asarray(res).flatten()
            res = (res - np.min(res)) / (np.max(res) - np.min(res) )
            res = np.asarray(res).reshape(-1,5)
            # coef = self.regression.reg.coef_
            coef = [-5, 4, 3, 2, 1]
            intercept = self.regression.reg.intercept_
            # res = tf.math.multiply(coef, tf.reshape(res,[-1,5]))
            user_vector = np.multiply(coef, truthy.reshape(-1,5))
            # truthy = truthy.reshape(1,-1)
            res = np.multiply(coef, res.reshape(-1,5))
            # truthy = np.add(np.multiply(coef, truthy), intercept)
            # res = res.reshape(1,-1)
            # print(np.asarray(res).shape)
            # print(truthy.shape)
            # print(np.asarray(res).flatten()[mask])
            # print(truthy[0][mask])
            # return (np.sum(np.abs(np.subtract(np.asarray(res).flatten()[mask], truthy[0][mask])))/int(len(indices*0.05)))**0.5
            poll_expectation = np.sum(res, axis=1)#.reshape(-1,self.num_engagements)), axis=1) # add specific engagement sum here
            true_expectation = np.sum(truthy, axis=1)

            timed_polls = range(len(self.polls))
            scores = []
            true_scores = []
            timed_scores = []
            tot_devs = []
            for i,idx in enumerate(poll_expectation):
                # if i in mask:
                    # Never recommend a poll that a user has already interacted with
                    if (self.polls[i].pollid not in self.users[userid].polls.keys()) or i in mask: #or self.polls[i].pollid == "-LVcLeir9mEFVXOqDd30"
                    # print("yay")
                        # tot_devs.append(poll_expectation[i]-np.mean(poll_expectation))
                        scores.append([self.poll_score(poll_expectation[i], self.polls[i]), self.polls[i].pollid, i])
                        true_scores.append([self.poll_score(true_expectation[i], self.polls[i]), self.polls[i].pollid, i])
                        # timed_scores.append([self.poll_score(timed_polls[i], self.polls[i]), self.polls[i].pollid, i])
            scores = sorted(scores, key=lambda x: x[0], reverse=True)
            # for idx,i in enumerate(scores):
            #     if i[1] == "-LVcLeir9mEFVXOqDd30":
            #         return idx
            # return 500
            true_scores = sorted(true_scores, key=lambda x: x[0], reverse=False)
            # timed_scores = sorted(timed_scores, key=lambda x: x[0], reverse=False)
            # return scores
            return [self.order_difference(true_scores, scores),self.order_difference(true_scores, timed_scores), 
            self.precision_at_50(true_scores, scores, mask), self.precision_at_50(true_scores, timed_scores, mask), np.mean(tot_devs)]
            # print(scores[:num_recs])

    def make_latent(self):
        arr = []
        u_vector_length = len(self.polls)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.saver.restore(sess, 'refactored/model.ckpt')
            for i in self.users.keys():
                user_vector = self.users[i].get_engagement_array(u_vector_length).reshape(-1,u_vector_length*self.num_engagements)

                arr.append(sess.run([self.encoder_op], feed_dict={self.X: user_vector}))
        return arr
        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     u_vector_length = len(self.polls)
        #     user_vector = self.users[userid].get_engagement_array(u_vector_length).reshape(-1,u_vector_length*self.num_engagements)
        #     self.saver.restore(sess, 'refactored/model.ckpt')
        #     res = sess.run([self.decoder_op], feed_dict={self.X: user_vector})
        #     res = np.asarray(res).reshape(-1,5)
        #     coef = self.regression.reg.coef_
        #     intercept = self.regression.reg.intercept_
        #     res = np.add(np.multiply(coef, res), intercept)
        #     poll_expectation = np.sum(res, axis=1)
        #     user_vector = np.add(np.multiply(coef, user_vector.reshape(-1,5)), intercept)
        #     true_expectation = np.sum(user_vector,axis=1)
        #     # poll_expectation = np.sum((np.asarray(res).reshape(-1,self.num_engagements)), axis=1) # add specific engagement sum here
        #     # true_expectation = np.sum((np.asarray(user_vector).reshape(-1,5)), axis=1)
        #     timed_polls = range(len(self.polls))
        #     scores = []
        #     true_scores = []
        #     timed_scores = []
        #     for i in np.argsort(-poll_expectation):
        #         # Never recommend a poll that a user has already interacted with
        #         if self.polls[i].pollid not in self.users[userid].polls.keys():
        #             scores.append([self.poll_score(poll_expectation[i], self.polls[i]), self.polls[i].pollid])
        #             true_scores.append([self.poll_score(true_expectation[i], self.polls[i]), self.polls[i].pollid])
        #             timed_scores.append([self.poll_score(timed_polls[i], self.polls[i]), self.polls[i].pollid])
        #     scores = sorted(scores, key=lambda x: x[0], reverse=True)
        #     true_scores = sorted(true_scores, key=lambda x: x[0], reverse=True)
        #     timed_scores = sorted(timed_scores, key=lambda x: x[0], reverse=True)
        #     return [self.order_difference(true_scores, scores),self.order_difference(true_scores, timed_scores), 
        #     self.precision_at_50(true_scores, scores), self.precision_at_50(true_scores, timed_scores)]

    def poll_score(self, poll_expectation, poll):
        # print(poll_expectation)
        # print(poll_expectation)
        return poll_expectation
        time_ago = (time.time() - poll.timestamp)
        return poll_expectation * np.exp(-time_ago/100)# / 1000)
        # time_ago = time_ago**(1/3)
        # return poll_expectation/(time_ago**1/10)
        
        # return poll_expectation + time_ago/200
        # # print(time_ago)
        # return ((poll_expectation*1000000000) / ((time_ago+1))**1) + poll.boost

    def order_difference(self,truelist, predictedlist):
        score = 0
        truelist = [x[1] for x in truelist]
        for idx,i in enumerate(predictedlist[:30]):
            score += np.absolute(truelist.index(i[1]) - idx)**2
        return np.sqrt(score/30)

    def precision_at_50(self, truelist, predictedlist, mask):
        p = 0 
        # truelist = [x[1] for x in truelist[:50]]
        for i in predictedlist[:50]:#[:len(mask)]:
            if i[2] in mask:
                p+=1
        if len(mask) > 0:
            return p/len(mask)
        else:
            return 0

    def new_poll(self,userid,pollid,timestamp,title,this,that):
        new_poll = poll.Poll(userid,pollid, timestamp, title, this, that)
        self.polls.append(new_poll)
        self.update_graph()
        self.add_interaction(userid,pollid,self.interaction_dict['owns'], self.polls.index(new_poll))
        
    def new_user(self, userid, lastActive):
        u = user.User(userid, lastActive)
        if userid not in self.users.keys():
            self.users[userid] = u

    def update_graph(self):

        # tf Graph input
        self.num_input = len(self.polls)*self.num_engagements

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.saver.restore(sess, 'refactored/model.ckpt')
            var = sess.run(self.weights1['encoder_h1'])
            new = np.zeros(shape=(self.num_engagements,self.num_hidden_1)).astype('float32')
            var = np.append(var, new, axis=0)
            self.weights1['encoder_h1'] = tf.Variable(var)
            print(self.weights1['encoder_h1'].shape)

            var = sess.run(self.weights1['decoder_h1'])
            new = np.full(shape=(self.num_engagements,self.num_hidden_1), fill_value=self.new_poll_weight).astype('float32')
            var = np.append(np.swapaxes(var,0,1), new, axis=0)
            self.weights1['decoder_h1'] = tf.Variable(np.swapaxes(var,0,1))

            # var = sess.run(self.biases1['encoder_b1'])
            # new = np.random.normal(1,size=(4,)).astype('float32')
            # var = np.append(var, new, axis=0)
            # self.biases1['encoder_b1'] = tf.Variable(var)

            var = sess.run(self.biases1['decoder_b1'])
            new = np.zeros(self.num_engagements).astype('float32')
            var = np.append(var, new, axis=0)
            self.biases1['decoder_b1'] = tf.Variable(var)

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
            # self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

            # Initialize the variables (i.e. assign their default value)
            self.init = tf.global_variables_initializer()
            self.saver.save(sess, 'refactored/model.ckpt')
        
        # self.saver = tf.train.Saver()

    # Batch add interactions in the form:
    # (userids:[1,2,3], pollids:[[32,2,4],[334,2],[2]], interactions:[[8,2],[4],[2]])
    def add_interaction(self, userids, pollids, interactions, pollidx):
        self.users[userids].add_interaction(pollids, interactions, pollidx)
        # for idx,i in enumerate(userids):
        #     for jdx,j in enumerate(pollids[idx]):
        #         for k in interactions[idx][jdx]:
        #             self.users[i].add_interaction(j, k, pollidx)
        self.interactions_counter += 1
        if self.interactions_counter % self.retrain_delay == 0:
            self.interactions_counter = 0 
            self.trainer(self.graph_update_epochs)


a = Autoencoder(num_epochs=5001, denoising=False, masking=0.5, display_step=200)
a.initial_trainer(save=True)
# a = Autoencoder(num_epochs=5000, denoising=False, masking=0.5)
# # to = 0
# # for i in a.test_users:
# #     to += i.num_interactions()
# # print(to/len(a.test_users))
# precs = []
# for i in range(0,5):
#     precs.append(a.initial_trainer(save=True))

# arr = a.make_latent()
# print(np.var(np.asarray(arr)))

# hidden_interactions = [[u'ssauRCxp4gMDGYm9w4jFwxWabvG2', 1], [u'HZ6ZcUVHeXYmZIxwwMrQm7WNqfH3', 1], [u'fChiYAtps9hy2pXgfRe8APAFgeT2', 1], [u'kEZfgYC7FRRLq78sa0C1oDpuPZ13', 1], [u'rv4PMYMToXN2oIoYoYcXs8k8kUC3', 1], [u'Vrv2bxkdZ4S19ECBQ7pWhrLkdRC2', 1], [u'o8BLoR68Ylgnk88rKlKd8uNhFN32', 1], [u'xAVTUQHIRNdXpryNn2yUfALbk892', 1], [u'UX0Os30KbXRAOTYSKs2BLGaz61k2', 1], [u'y0RqzcIEycRGsEJog7z8SbbvNb13', 1], [u'REnpwB7yGDe1whBlwqfAfuIGyWS2', 1], [u'SJYZdf8Vk4PNoWA7gm3Cwe62IwC3', 1], [u'bUgIR9tAJGgmlxCkfVBgbM41V7B2', 1], [u'dV7q7Blcpjd4IBRUhCIr1FbBAlA3', 1], [u'ut2a2ZN37zRNgePREeNde2a79bE3', 1], [u'G5fXwWOC01ccMVvEiSTZ9BsJOH13', 1], [u'L7S2I2u5psfTDqEA30aDSWE7h2g2', 16], [u'UknrMnapbFcaT6DMs2OSaJjrKZ72', 1], [u'aDPjMUajnDQIvNdGpGh9gmFRA2k2', 1], [u'A3cxNwOMssPsKw0RmcS3QqYGLoj1', 1], [u'krO9dUgbEJeavIjlxTUSgd9Hef62', 1], [u'7dBSwgcgirgaBKXD73Dh075zKel1', 1], [u'eaxElYr0iKfWNtrVmcren7OEaDw1', 1], [u'F4ofvTitUzfBHJbbgPgjQH7x1ko2', 16], [u't7ov5qEFNPOPgpkJiMjHgvh3IyX2', 1], [u'qzM8Kp884JchdSmc9zXkuGWkwFm1', 1], [u'bavjkXVtDIbPTFNhKdXQWNxp7s43', 1], [u'MV5fbiGluYWaR5KJnBPXwyByc9l1', 16], [u'AWRZpNVejxMzmNXcuMBBEWg36FJ2', 1], [u'v2Y1q8B9wrO9SknQzTvKi1aTUXc2', 1], [u'3sAqU9MhkkdFd8zxNlXCdmJd4Lu1', 1], [u'5PLvJeJttUasuJFM5vXAkh8sa5G2', 1], [u'NZp7LGUvKLZPoEvvLSqnAxYFG7t1', 1], [u'SodaZQhSL9UjxJ4vRDeimlG8xAK2', 1], [u'YnkBuWeuZlZk6tvMtCfDZ5ac2RJ3', 1], [u'cR5YLdfFBEYNgyYnp7JfFmlpGM92', 1], [u'45f5hYDoeNY9MSSONZ0tlHt8pbr2', 1], [u'RRV1VIDtkET6ovdAtUHPZQVs1RP2', 1], [u'gv3YXCU3nebRft1DDXOkeh7g09g1', 1], [u'wr2Tf0DrY9duPHZEz687BoG84Bi1', 1], [u'mOSXPOgyxUTjUCA3x0kp59GE2Sm2', 1], [u'ZjuFGLqUuxdXQPIkSg6smIsYrMI3', 1], [u'z6kKIG0L7ccNP6cg9dEEAnsRFgk1', 1], [u'hoLXtDQpmBWqTI0liRCoWfHx4tF2', 1], [u'cjasuQmHhBbzaxih4LYpOB9WXRt2', 1], [u'DNR8idL2NiPv9mP4JjwmYekQyBF2', 1], [u'uCIUbhJ4UFheU7okKjlHo3Y1Hma2', 1], [u'L0N9vgBLAlTKOk24pQ2BCoVMmr12', 1], [u'ue5fRVG1seQvpukudDouMzrL8Gp2', 1], [u'6aoC81SXC2YsLsBoT1aLr2BvK993', 1], [u'Y4tgpPHQdTcLQ1NLOAal1K90ftx1', 1], [u'3W4pzU6zUVapSthzkG498cC2hbM2', 1], [u'PkEnZ9IJuMcK0k7hjr5aZkHZCrq2', 1], [u'Cv2fTDo9haTgrpYG2fkKFWSYWXA2', 16], [u'isFsisiQFzY8UXO8EgK6Bbzwp4n2', 1], [u'hffP2jBG2bPnrmNhAXFOEwQEpjD2', 1], [u'xXxm8kENK9elP8HdkrBomWJjSnH2', 1], [u'CnhrSwhP8GXdZazlrSrerFAURxz1', 1], [u'NowgqkLsAwcJiCWbUHw41osoXUE3', 1], [u'Bbd8BqEtqyTlkZWdJZ0Yhv6hApp1', 1], [u'vFvM2fv7OFhoH9iH4QRBHnITpBX2', 1], [u'nSoDWSANS5VZGtmmONCVS02nT8n1', 16], [u'qtHQrAdTJyO1Mw0GP6lygfj0e0j1', 1], [u'uGeX3IbQ1ifSkpePtuHGXx2ZwW83', 1], [u'WmlVl2vAwTYlNr387rhfQDTY2S63', 1], [u'thGGbLDwftgvKZfwQz0RuCr0ufU2', 1], [u'58tnxZ2MdRfEOxpWTsTVKyr9ryw1', 1], [u'tzIsQULp22W1Hsmv2WED6TIdFMJ3', 1], [u'SDZXp0np8fSPWvi4GQ4RKsrH2dL2', 1], [u'fc4IbdViIXPvqXTpt6LxWctbXwn2', 1], [u'TIvCzYKDLtaxLUAF1hzs44HnJ5Z2', 1], [u'UFTzdO1OBDV1WkNtaIEf5umcz4E2', 1], [u'xC46hES2oxfOIA7eSbYH0z4zThe2', 1], [u'ZEzJAjZ2miPeBSOPkDLtmL0sTCn2', 1], [u'Pktw7H1sWDPzUzQnAC4Ui9zQIm63', 1], [u'Mew5o6fgpFU6w7AWwEHLoHcTViA3', 1], [u'J0MM8WIQvtXpu7b0EFKiQYmuBWL2', 1], [u'xC46hES2oxfOIA7eSbYH0z4zThe2', 4], [u'PkEnZ9IJuMcK0k7hjr5aZkHZCrq2', 4], [u'v2Y1q8B9wrO9SknQzTvKi1aTUXc2', 4], [u'cR5YLdfFBEYNgyYnp7JfFmlpGM92', 4], [u'5PLvJeJttUasuJFM5vXAkh8sa5G2', 4], [u'G5fXwWOC01ccMVvEiSTZ9BsJOH13', 4], [u'hffP2jBG2bPnrmNhAXFOEwQEpjD2', 4], [u'isFsisiQFzY8UXO8EgK6Bbzwp4n2', 4], [u'qzM8Kp884JchdSmc9zXkuGWkwFm1', 4], [u'CnhrSwhP8GXdZazlrSrerFAURxz1', 4], [u'Mew5o6fgpFU6w7AWwEHLoHcTViA3', 4], [u'z6kKIG0L7ccNP6cg9dEEAnsRFgk1', 4], [u'DNR8idL2NiPv9mP4JjwmYekQyBF2', 4], [u'uGeX3IbQ1ifSkpePtuHGXx2ZwW83', 4], [u'HZ6ZcUVHeXYmZIxwwMrQm7WNqfH3', 4], [u'L0N9vgBLAlTKOk24pQ2BCoVMmr12', 4]]
# # hidden_users = [x[0] for x in hidden_interactions]
# a = Autoencoder()
# a.new_poll(u'hnZIsnWtxUWSzPGmjezjDzyCrj02',"-LVcLeir9mEFVXOqDd30",time.time()-3840,"testing","testing","123")
# arr = []
# for jdx,j in enumerate(hidden_interactions[::5]):
#     pos = []
#     for i in hidden_interactions:
#         n = a.predict(i[0])
#         print(n)
#         pos.append(n)
#     print("-"*30)
#     print(np.mean(pos))
#     print("-"*30)
#     arr.append(np.mean(pos))
#     a.add_interaction(hidden_interactions[jdx][0],"-LVcLeir9mEFVXOqDd30",hidden_interactions[jdx][1], len(a.polls)-1)
#     a.add_interaction(hidden_interactions[jdx+1][0],"-LVcLeir9mEFVXOqDd30",hidden_interactions[jdx+1][1], len(a.polls)-1)
#     a.add_interaction(hidden_interactions[jdx+2][0],"-LVcLeir9mEFVXOqDd30",hidden_interactions[jdx+2][1], len(a.polls)-1)
#     a.add_interaction(hidden_interactions[jdx+3][0],"-LVcLeir9mEFVXOqDd30",hidden_interactions[jdx+3][1], len(a.polls)-1)
#     a.add_interaction(hidden_interactions[jdx+4][0],"-LVcLeir9mEFVXOqDd30",hidden_interactions[jdx+4][1], len(a.polls)-1)
#     hidden_interactions = hidden_interactions[5:]
# print(arr)




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


# total_interactions = 0
# total_votes = 0
# av_pos = []
# a = Autoencoder(num_epochs=5000, new_poll_weight=0.0075)
# random_submitter = a.users.keys()[np.random.randint(len(a.users))]
# for k in [60, 120, 240, 480, 960, 1920, 3840, 7680, 15360, 37200, 74400][::-1]:
#     a.new_poll(a.users[random_submitter].userid, "test", time.time()-k, "test", "this", "that")
#     pos = 0
#     for j in a.test_users[:5]:
#         for idx,i in enumerate(a.predict(j.userid)):
#             if i[1] == "test":
#                 pos += idx
#                 break
#     av_pos.append(pos/5)
#     print(pos/5)
# # print(np.mean(np.asarray(av_pos).reshape(10,-1), axis=1))
#         # print(pos/len(a.test_users)) 
# print(av_pos)
# # a.initial_trainer()
# for i in a.users.keys():
#     total_interactions += len(np.argwhere(a.users[i].get_engagement_array(len(a.polls)) > 0))
#     total_votes += a.users[i].num_votes()


# print(float(total_interactions)/len(a.users.keys()))
# print(len(a.data_provider.polls))
# print(len(a.users))








