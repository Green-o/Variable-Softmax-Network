import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

neuron_seq = [784]
batch_size = 100

# Input and output values 
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32)

# Network population from CLI input
while True:
    try:
        layer_count = int(input("Hidden layer count: "))
    except ValueError:
        print("Non-integer input.")
    else:
        break

while len(neuron_seq)-1 < layer_count:
    while True:
        try:
            inp = int(input("HL %s neuron count: " % len(neuron_seq)))
        except ValueError:
            print("Non-integer input.")
        else:
            break
    neuron_seq.append(inp)
neuron_seq.append(10)

train_steps = input("Train iterations: ")

# Create hidden layers using input data
layers = []
for l in range(len(neuron_seq)-1):
    layer = {'weights': tf.Variable(tf.truncated_normal([neuron_seq[l], neuron_seq[l+1]], stddev=.05)),
             'biases': tf.Variable(tf.constant(0.05, shape=[neuron_seq[l+1]]))}
    layers.append(layer)

# Run computational graph
def neural_network_model(data):
    ops = []
    for l in range(len(layers)):
        if l == 0:
            op = tf.add(tf.matmul(data, layers[l]['weights']), layers[l]['biases'])
        else:
            op = tf.add(tf.matmul(ops[l-1], layers[l]['weights']), layers[l]['biases'])
            if l < len(layers)-1:
                op = tf.nn.relu(op)
        ops.append(op)

    return ops[len(ops)-1]

# Train and update user on network's process
def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    print("v-Training network-v")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(train_steps+1):
            iter_x, iter_y = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: iter_x, y: iter_y})

            if step % 100 == 0 or step == train_steps:
                correct = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
                accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
                step_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
                if step % 100 == 0:
                    print("%s/%s steps - %.3f%% accuracy" % (step, train_steps, step_accuracy*100))
                if step == train_steps:
                    print("Final accuracy: %.3f%%" % (step_accuracy*100))

train_neural_network(x)
input("Press enter to close")
