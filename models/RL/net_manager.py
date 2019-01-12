import tensorflow as tf

from models.RL.cnn import CNN


class NetManager():
    def __init__(self, num_input, num_classes, learning_rate, data_set,
                 batch_size=100,
                 dropout_rate=0.85):

        self.num_input = num_input
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.x_train, self.y_train, self.x_test, self.y_test = data_set

        self.epochs = 3
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate

    def get_reward(self, action, step, pre_acc):
        action = [action[0][0][x:x + 4] for x in range(0, len(action[0][0]), 4)]
        cnn_drop_rate = [c[3] for c in action]
        with tf.Graph().as_default() as g:
            with g.container('experiment' + str(step)):
                model = CNN(self.num_input, self.num_classes, action)
                loss_op = tf.reduce_mean(model.loss)
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                train_op = optimizer.minimize(loss_op)

                with tf.Session() as train_sess:
                    init = tf.global_variables_initializer()
                    train_sess.run(init)
                    for epoch in range(self.epochs):
                        batches = len(self.x_train) // self.batch_size
                        for step in range(batches):
                            batch_x = self.x_train[step * self.batch_size:(step + 1) * self.batch_size]
                            batch_x = batch_x.reshape((-1, 784))
                            batch_y = self.y_train[step * self.batch_size:(step + 1) * self.batch_size]
                            feed = {model.X: batch_x,
                                    model.Y: batch_y,
                                    model.dropout_keep_prob: self.dropout_rate,
                                    model.cnn_dropout_rates: cnn_drop_rate}
                            _ = train_sess.run(train_op, feed_dict=feed)

                            if step % 100 == 0:
                                # Calculate batch loss and accuracy
                                loss, acc = train_sess.run(
                                    [loss_op, model.accuracy],
                                    feed_dict={model.X: batch_x,
                                               model.Y: batch_y,
                                               model.dropout_keep_prob: 1.0,
                                               model.cnn_dropout_rates: [1.0] * len(cnn_drop_rate)})
                                print("Step " + str(step) +
                                      ", Minibatch Loss= " + "{:.4f}".format(loss) +
                                      ", Current accuracy= " + "{:.3f}".format(acc))
                    batch_x, batch_y = self.x_test, self.y_test
                    batch_x = batch_x.reshape((-1, 784))
                    loss, acc = train_sess.run(
                        [loss_op, model.accuracy],
                        feed_dict={model.X: batch_x,
                                   model.Y: batch_y,
                                   model.dropout_keep_prob: 1.0,
                                   model.cnn_dropout_rates: [1.0] * len(cnn_drop_rate)})
                    print("!!!!!!acc:", acc, pre_acc)
                    if acc - pre_acc <= 0.01:
                        return acc, acc
                    else:
                        return 0.01, acc
