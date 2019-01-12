import tensorflow as tf
import argparse
from models.RL.cnn import CNN
from utils.data_loader import load_data


def main(action, name):
    x_train, y_train, x_test, y_test = load_data()
    action = [int(x) for x in action.split(",")]
    training_epochs = 10
    batch_size = 100

    action = [action[x:x + 4] for x in range(0, len(action), 4)]
    cnn_drop_rate = [c[3] for c in action]

    model = CNN(784, 10, action)
    loss_op = tf.reduce_mean(model.loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    train_op = optimizer.minimize(loss_op)

    tf.summary.scalar('acc', model.accuracy)
    tf.summary.scalar('loss', tf.reduce_mean(model.loss))
    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(name, graph=tf.get_default_graph())

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for epoch in range(training_epochs):
        for step in range(int(len(x_train) / batch_size)):
            batch_x = x_train[step * batch_size:(step + 1) * batch_size]
            batch_x = batch_x.reshape((-1, 784))
            batch_y = y_train[step * batch_size:(step + 1) * batch_size]
            feed = {model.X: batch_x,
                    model.Y: batch_y,
                    model.dropout_keep_prob: 0.85,
                    model.cnn_dropout_rates: cnn_drop_rate}
            _, summary = sess.run([train_op, merged_summary_op], feed_dict=feed)
            summary_writer.add_summary(summary, step + (epoch + 1) * int(len(x_train) / batch_size))

        print("epoch: ", epoch + 1, " of ", training_epochs)

        batch_x, batch_y = x_test, y_test
        batch_x = batch_x.reshape((-1, 784))
        loss, acc = sess.run(
            [loss_op, model.accuracy],
            feed_dict={model.X: batch_x,
                       model.Y: batch_y,
                       model.dropout_keep_prob: 1.0,
                       model.cnn_dropout_rates: [1.0] * len(cnn_drop_rate)})

        print("Network accuracy =", acc, " loss =", loss)
    print("Final accuracy for", name, " =", acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', default="5, 32, 2,  5, 3, 64, 2, 3")
    parser.add_argument('--name', default="model")
    args = parser.parse_args()

    main(args.architecture, args.name)
