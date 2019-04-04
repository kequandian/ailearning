"""使用TensorFlow代码生成PROJECTOR所需要的日志文件来可视化MNIST测试数据在最后的输出层向量"""

import tensorflow as tf
import mnist_inference
import os
# 加载用于生成projector日志的帮助函数
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data

# 定义训练模型需要的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 10000          # 可以通过调整这个参数来控制训练迭代轮数
MOVING_AVERAGE_DECAY = 0.99

# 和日志文件相关的文件名和目录地址
LOG_DIR = 'log'
SPRITE_FILE = 'mnist_sprite.png'
META_FILE = "mnist_meta.tsv"
TENSOR_NAME = "FINAL_LOGITS"


def train(mnist):
    # 输入数据的命名空间
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 将处理滑动平均相关的计算都放在moving_average的命名空间下
    with tf.name_scope("moving_average"):
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.name_scope("loss_function"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    with tf.name_scope("train_step"):
        learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,
            staircase=True)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        with tf.control_dependencies([train_step, variables_averages_op]):
            train_op = tf.no_op(name='train')

    # 训练模型。
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run(
                [train_op, loss, global_step], feed_dict={x: xs, y_: ys})

            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (i, loss_value))

        # 计算MNIST测试数据对应的输出层矩阵
        final_result = sess.run(y, feed_dict={x: mnist.test.images})

    # 返回输出层矩阵的值
    return final_result


# 生成可视化最终输出层向量所需要的日志文件
def visualisation(final_result):
    # 使用一个新的变量来保存最终输出层向量的结果
    y = tf.Variable(final_result, name=TENSOR_NAME)
    summary_writer = tf.summary.FileWriter(LOG_DIR)

    # 通过projector.ProjectorConfig类来帮助生成日志文件
    config = projector.ProjectorConfig()
    # 增加一个需要可视化的embedding结果
    embedding = config.embeddings.add()
    # 制定这个embedding结果对应的TensorFlow变量名称
    embedding.tensor_name = y.name

    # 指定embedding结果所对应的原始数据信息
    embedding.metadata_path = META_FILE
    # 指定sprite图像
    embedding.sprite.image_path = SPRITE_FILE
    # 指定单张图片的大小，用于从sprite图像中截取正确的原始图片
    embedding.sprite.single_image_dim.extend([28, 28])

    # 将PROJECTOR所需要的内容写入日志文件
    projector.visualize_embeddings(summary_writer, config)

    # 生成会话，初始化新声明的变量并将需要的日志信息写入文件
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(LOG_DIR, "model"), TRAINING_STEPS)

    summary_writer.close()


# 主函数先调用模型训练的过程，再使用训练好的模型来处理MNIST测试数据，最后将得到的输出层矩阵输出到PROJRCTOR需要的日志文件中
def main(argv_None):
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    final_result = train(mnist)
    visualisation(final_result)


if __name__ == '__main__':
    tf.app.run()
