import math
import tensorflow as tf
from tensorflow.python.framework import graph_util

def buildFC_AEDynamic(shape=(64, 64, 3), embedding_size=256, num_convs=3, stride=(2, 2), ksize=(2, 2)):


    assert embedding_size % (shape[0] / (stride[0]**num_convs) * shape[1] / (stride[1]**num_convs)) == 0 and num_convs > 1

    num_fil = embedding_size / (shape[0] / (stride[0]**num_convs) * shape[1] / (stride[1]**num_convs))

    # NAME = inputs for export
    x = tf.placeholder(
        tf.float32, [None, shape[0], shape[1], shape[2]], name='inputs')

    # Xavier weight initialization for convolutional filters
    def xavier_conv_init(shape):
        return tf.random_normal(shape, stddev=math.sqrt(1.0 / (shape[0] * shape[1] * shape[2])))

    # Xavier weight initialization for convolutional filters
    def xavier_fc_init(shape):
        return tf.random_normal(shape, stddev=math.sqrt(1.0 / shape[0]))

    def addConv(layer, stride, kernel, filters, filers_last):
        return tf.nn.relu(tf.nn.conv2d(layer, tf.Variable(xavier_conv_init((kernel[0], kernel[1], int(filers_last), int(filters)))), 
            strides=[1, stride[0], stride[1], 1], padding='SAME'))

    def addDeconv(layer, stride, kernel, filters, filers_last,  depth, name=None):
        return tf.nn.relu(tf.nn.conv2d_transpose(layer, tf.Variable(xavier_conv_init((kernel[0], kernel[1], int(filters), int(filers_last)))),
                                                [tf.shape(x)[0], int(shape[0]/(stride[0]**depth)), int(shape[1]/(stride[1]**depth)), int(filters)],
                                                [1, stride[0], stride[1], 1],
                                                padding='SAME'), name=name)


    cur_layer = addConv(x, stride, ksize, 2 ** 5, shape[2])
    for i in range(num_convs-2):
        cur_layer = addConv(cur_layer, stride, ksize, 2**(6+i), 2**(5+i))
    cur_layer = addConv(cur_layer, stride, ksize, num_fil, 2**(5+num_convs-2))

    # NAME = latent for export
    z = tf.reshape(cur_layer, [-1, int(shape[0] / (stride[0]**num_convs) * shape[1] / (stride[1]**num_convs) * num_fil)], name="latent")

    cur_layer = addDeconv(cur_layer, stride, ksize, 2 ** (5 + num_convs - 2), num_fil, num_convs-1)
    for i in range(num_convs-2):
        cur_layer = addDeconv(cur_layer, stride, ksize, 2**(5+num_convs-3-i), 2**(5+num_convs-2-i), num_convs-2-i)

    # NAME = reconsturction for export
    y = addDeconv(cur_layer, stride, ksize, shape[2], 2 ** 5, 0, name="reconstruction")

    # cost function measures pixel-wise difference
    cost = tf.reduce_mean(tf.squared_difference(x, y))

    return x, y, z, cost

def main():

    sess = tf.Session()

    data_shape = (64,64,3)
    _, _, _, _ = buildFC_AEDynamic(shape=data_shape, embedding_size=128, num_convs=3, stride=(2, 2), ksize=(4, 4))

    sess.run(tf.global_variables_initializer())

    output_graph_file = "checkpoints/cae.pb"

    print('Exporting trained model to', output_graph_file)
    output_node_names = "inputs,reconstruction,latent"
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, sess.graph_def, output_node_names.split(","))

    with tf.gfile.GFile(output_graph_file, "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("saved: cnn")
    print("%d ops in the final graph." % len(output_graph_def.node))


if __name__ == "__main__":
    main()
