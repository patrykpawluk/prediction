import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

dataset = pd.read_csv('./datasets/credit_card_frauds.csv',delimiter=",")
X, Y = dataset.iloc[:,:-1], dataset.iloc[:,-1]

X_train1, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=123)

min_max_scaler = preprocessing.MinMaxScaler()
X_train = pd.DataFrame(min_max_scaler.fit_transform(X_train1))

#print(X_train.size)

n_nodes_hl1 = 100
n_nodes_hl2 = 100
n_nodes_hl3 = 100

n_classes = 2
x_var = 29
batch_size = 128

x = tf.placeholder(tf.float32, shape=(None, x_var))
y = tf.placeholder(tf.float32, shape=(None, None))

# height x width
def preprocess(X, Y):
  X = tf.cast(X,tf.float32)
  Y = tf.cast(Y,tf.float32)

  return X, Y

def create_dataset(xs, ys, n_classes=2):
  yss = tf.one_hot(ys, depth=n_classes)
  return tf.data.Dataset.from_tensor_slices((xs, yss)) \
    .map(preprocess) \
    .shuffle(len(ys)) \
    .batch(batch_size)

train_dataset = create_dataset(X_train, y_train)
test_dataset = create_dataset(X_test, y_test)

print(train_dataset.take(batch_size))

def neural_network_model(data):
    
    # (input_data * weights) + biases

    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([x_var, n_nodes_hl1])),
                    'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                    'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
                    
    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                    'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes]))}
    
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights'], transpose_a=False), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights'], transpose_a=False), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights'], transpose_b=False), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights'], transpose_b=False) + output_layer['biases']
    #logit = tf.cast(tf.arg_max(output, 1), tf.float32)

    return output


#@tf.function
def train_neural_network(x):

    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = prediction, labels = y))
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    hm_epochs = 10

    iter = train_dataset.make_initializable_iterator()
    el = iter.get_next()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            sess.run(iter.initializer)
            for _ in range(int(len(X_train)/batch_size)):
                epoch_x, epoch_y = sess.run(el)
                _, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float32'))

        y_train_hot = tf.one_hot(y_train, depth=2)
        y_train_hot_np = sess.run(y_train_hot)

        print('Accuracy:',accuracy.eval({x:X_train, y:y_train_hot_np}))
        
train_neural_network(x)