from sys import platform

import numpy as np
import math
import cv2

import tensorflow as tf
from tensorflow.python.tools import freeze_graph

import h5py

def augment_patch(patch):

    flip = np.random.randint(0, 3)
    if flip > 0:
        patch = cv2.flip(patch, flip-2)

    swap = np.random.randint(0, 6)
    if swap == 1:
        patch[:, :, 0:3] = 1.0 - patch[:, :, 0:3]   # Invert colors
    elif swap == 2:
        patch[:, :, 0:3] = patch[:, :, [0, 2, 1]]
    elif swap == 3:
        patch[:, :, 0:3] = patch[:, :, [2, 0, 1]]
    elif swap == 4:
        patch[:, :, 0:3] = patch[:, :, [1, 0, 2]]
    elif swap == 5:
        patch[:, :, 0:3] = patch[:, :, [1, 2, 0]]


def buildFC_AEDynamic(shape=(64, 64, 3), embedding_size=256, num_convs=3, stride=(2, 2), ksize=(2, 2)):
    assert embedding_size % (shape[0] / (stride[0]**num_convs) * shape[1] / (stride[1]**num_convs)) == 0 and num_convs > 1
    num_fil = embedding_size / (shape[0] / (stride[0]**num_convs) * shape[1] / (stride[1]**num_convs))

    x = tf.placeholder(
        tf.float32, [None, shape[0], shape[1], shape[2]], name='x')

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

    z = tf.reshape(cur_layer, [-1, int(shape[0] / (stride[0]**num_convs) * shape[1] / (stride[1]**num_convs) * num_fil)], name="z")

    cur_layer = addDeconv(cur_layer, stride, ksize, 2 ** (5 + num_convs - 2), num_fil, num_convs-1)
    for i in range(num_convs-2):
        cur_layer = addDeconv(cur_layer, stride, ksize, 2**(5+num_convs-3-i), 2**(5+num_convs-2-i), num_convs-2-i)
    y = addDeconv(cur_layer, stride, ksize, shape[2], 2 ** 5, 0, name="y")

    # cost function measures pixel-wise difference
    cost = tf.reduce_mean(tf.squared_difference(x, y))

    return {'x': x, 'z': z, 'y': y, 'cost': cost}

def main():
    ### load your data as numpy array [num_samples, height, width, channels]
    data = np.copy(h5py.File("auto.h5", 'r')["/data"]) 

    batch_size = 100
    feat_dim = 128

    data_shape = (data.shape[1], data.shape[2], data.shape[3])

    cae = buildFC_AEDynamic(shape=data_shape, embedding_size=feat_dim, num_convs=3, stride=(2, 2), ksize=(4, 4))

    optimizer = tf.train.AdamOptimizer(0.0001).minimize(cae['cost'])

    modelname = "cae_" + str(feat_dim)
    checkpoint_path = "checkpoints/"

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver()

        for epoch in range(3):

            print("Starting training epoch ", epoch)
            for batch in range(data.shape[0] // batch_size):
                position = batch * batch_size
                curr_dat = np.copy(data[position:position + batch_size])

                for patch in curr_dat:
                    augment_patch(patch)
                _, error, y = sess.run([optimizer, cae['cost'], cae['y']], feed_dict={cae['x']: curr_dat})

                print("Batch:", batch, "  Loss: ", error)

                # Visualize the reconstruction of the first sample in batch
                cv2.imshow('Original', show_rgbd_patch(curr_dat[0]))
                cv2.imshow('Reconstruction', show_rgbd_patch(y[0]))
                cv2.waitKey(50)

            weights_file = saver.save(sess, checkpoint_path + modelname + ".ckpt")   # Write checkpoint with weights

        input_graph_file = checkpoint_path + modelname + ".pb.empty"
        output_graph_file = modelname + ".pb"

        print('Exporting trained model to', output_graph_file)
        tf.train.write_graph(sess.graph_def, ".", input_graph_file)     # Write graph definition

        # Freeze into one common file
        output_node_names = "x,y,z"
        freeze_graph.freeze_graph(input_graph_file, "", False, weights_file, output_node_names,
                                  "save/restore_all", "save/Const:0", output_graph_file, True, "")


if __name__ == "__main__":
    main()
