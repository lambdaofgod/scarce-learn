import numpy as np
import tensorflow as tf
from tqdm import trange


class ProtoNetTF:

    def __init__(self, im_height, im_width, channels, h_dim=64, z_dim=64, dropout_rate=0.1, scope_name_suffix='_0', sess=None):
        self.im_height = im_height
        self.im_width = im_width
        self.channels = channels
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.dropout_rate = dropout_rate
        self.scope_name_suffix = scope_name_suffix
        if sess is None:
            self.sess = tf.InteractiveSession()
        else:
            self.sess = sess

    def train(self, train_dataset, n_epochs, n_episodes, n_way, n_shot, n_query):
        n_classes, n_examples, im_width, im_height = train_dataset.shape

        x, q, y, training = setup_inputs(self.im_height, self.im_width, self.channels)
        self.x, self.q, self.y, self.training = x, q, y, training

        ce_loss, acc = setup_outputs(x, q, y, training, self.h_dim, self.z_dim, self.channels, self.dropout_rate)
        self.ce_loss, self.acc = ce_loss, acc

        train_op = setup_train_op(
            ce_loss,
            epoch_decay_interval=20,
            n_episodes=n_episodes)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        train_losses = []
        train_accs = []

        for ep in range(n_epochs):
            for epi in trange(n_episodes):
                epi_classes = np.random.permutation(n_classes)[:n_way]
                support = np.zeros([n_way, n_shot, im_height, im_width], dtype=np.float32)
                query = np.zeros([n_way, n_query, im_height, im_width], dtype=np.float32)

                for i, epi_cls in enumerate(epi_classes):
                    selected = np.random.permutation(n_examples)[:n_shot + n_query]
                    support[i] = train_dataset[epi_cls, selected[:n_shot]]
                    query[i] = train_dataset[epi_cls, selected[n_shot:]]

                support = np.expand_dims(support, axis=-1)
                query = np.expand_dims(query, axis=-1)
                labels = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_query)).astype(np.uint8)
                __, ls, ac = self.sess.run(
                    [train_op, ce_loss, acc],
                    feed_dict={x: support, q: query, y:labels, training: True}
                )
                train_losses.append(ls)
                train_accs.append(ac)
            print('[epoch {}/{}] => loss: {:.5f}, acc: {:.5f}'.format(ep+1, n_epochs, ls, ac))
        return train_losses, train_accs

    def test(self, test_dataset, n_test_classes, n_test_episodes, n_test_way, n_test_shot, n_test_query):
        n_classes, n_examples, im_width, im_height = test_dataset.shape

        print('Testing...')
        test_losses, test_accs = [], []

        for epi in range(n_test_episodes):
            epi_classes = np.random.permutation(n_test_classes)[:n_test_way]
            support = np.zeros([n_test_way, n_test_shot, im_height, im_width], dtype=np.float32)
            query = np.zeros([n_test_way, n_test_query, im_height, im_width], dtype=np.float32)

            for i, epi_cls in enumerate(epi_classes):
                selected = np.random.permutation(n_examples)[:n_test_shot + n_test_query]
                support[i] = test_dataset[epi_cls, selected[:n_test_shot]]
                query[i] = test_dataset[epi_cls, selected[n_test_shot:]]
            support = np.expand_dims(support, axis=-1)
            query = np.expand_dims(query, axis=-1)
            labels = np.tile(np.arange(n_test_way)[:, np.newaxis], (1, n_test_query)).astype(np.uint8)
            ls, ac = self.sess.run(
                [self.ce_loss, self.acc],
                feed_dict={self.x: support, self.q: query, self.y:labels, self.training:False}
            )
            test_accs.append(ac)
            test_losses.append(ls)
            if (epi+1) % 25 == 0:
                print('[test episode {}/{}] => loss: {:.5f}, acc: {:.5f}'.format(epi+1, n_test_episodes, ls, ac))

        avg_acc = (sum(test_accs) / len(test_accs))
        print('Average Test Accuracy: {:.5f}'.format(avg_acc))
        return test_losses, test_accs


def conv_block(inputs, out_channels, training, rate, name='conv'):
    with tf.variable_scope(name):
        conv = tf.layers.conv2d(inputs, out_channels, kernel_size=3, padding='SAME')
        conv = tf.contrib.layers.batch_norm(conv, updates_collections=None, decay=0.99, scale=True, center=True)
        conv = tf.nn.relu(conv)
        conv = tf.contrib.layers.max_pool2d(conv, 2)
        conv = tf.layers.dropout(conv, rate=rate, training=training)
        return conv


def encoder(x, h_dim, z_dim, training, rate, reuse=False):
    with tf.variable_scope('encoder', reuse=reuse):
        net = conv_block(x, h_dim, name='conv_1', training=training, rate=rate)
        net = conv_block(net, h_dim, name='conv_2', training=training, rate=rate)
        net = conv_block(net, h_dim, name='conv_3', training=training, rate=rate)
        net = conv_block(net, z_dim, name='conv_4', training=training, rate=rate)
        net = tf.contrib.layers.flatten(net)
        return net


def euclidean_distance(a, b):
    # a.shape = N x D
    # b.shape = M x D
    N, D = tf.shape(a)[0], tf.shape(a)[1]
    M = tf.shape(b)[0]
    a = tf.tile(tf.expand_dims(a, axis=1), (1, M, 1))
    b = tf.tile(tf.expand_dims(b, axis=0), (N, 1, 1))
    return tf.reduce_mean(tf.square(a - b), axis=2)


def setup_inputs(im_height, im_width, channels):
    x = tf.placeholder(tf.float32, [None, None, im_height, im_width, channels])
    q = tf.placeholder(tf.float32, [None, None, im_height, im_width, channels])
    y = tf.placeholder(tf.int64, [None, None])
    training = tf.placeholder(tf.bool)
    return x, q, y, training


def setup_outputs(x, q, y, training, h_dim, z_dim, channels, dropout_rate):
    x_shape = tf.shape(x)
    q_shape = tf.shape(q)

    num_classes, num_support = x_shape[0], x_shape[1]
    num_queries = q_shape[1]
    im_height = x_shape[2]
    im_width = x_shape[3]

    y_one_hot = tf.one_hot(y, depth=num_classes)

    emb_x = encoder(
        tf.reshape(x, [num_classes * num_support, im_height, im_width, channels]),
        h_dim,
        z_dim,
        training=training,
        rate=dropout_rate
    )
    emb_dim = tf.shape(emb_x)[-1]
    emb_x = tf.reduce_mean(tf.reshape(emb_x, [num_classes, num_support, emb_dim]), axis=1)
    emb_q = encoder(
        tf.reshape(q, [num_classes * num_queries, im_height, im_width, channels]),
        h_dim,
        z_dim,
        reuse=True,
        training=training,
        rate=dropout_rate
    )

    dists = euclidean_distance(emb_q, emb_x)
    log_p_y = tf.reshape(tf.nn.log_softmax(-dists), [num_classes, num_queries, -1])

    ce_loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_one_hot, log_p_y), axis=-1), [-1]))
    acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(log_p_y, axis=-1), y)))
    return ce_loss, acc


def setup_train_op(
        loss,
        starter_learning_rate=0.002,
        decay_rate=0.5,
        epoch_decay_interval=20,
        n_episodes=100):
    """
      Defaults are set according to original paper
    """

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        starter_learning_rate,
        global_step,
        epoch_decay_interval * n_episodes,
        decay_rate,
        staircase=True
    )

    return (
        tf.train.AdamOptimizer(learning_rate)
            .minimize(loss, global_step=global_step)
    )
