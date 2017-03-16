import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
from keras.utils import np_utils

import PIL.Image
from IPython.display import Image, display

# Convolutional Layer 1.
filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16         # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 36         # There are 36 of these filters.
num_classes = 10
# Fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected layer.



from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)

X_train = np.empty_like (data.train.images)
X_train[:] = data.train.images
Y_train = np.empty_like (data.train.labels)
Y_train[:] = data.train.labels
Y_train_label=[np.argmax(x) for x in Y_train]
Y_train_label_np=np.array(Y_train_label)
X_train_lt5=X_train[Y_train_label_np<5]
Y_train_lt5_label=Y_train_label_np[Y_train_label_np<5]
X_train_gt5=X_train[Y_train_label_np>=5]
Y_train_gt5_label=Y_train_label_np[Y_train_label_np>=5]
Y_train_lt5 = np_utils.to_categorical(Y_train_lt5_label, 5)



X_train_1 = X_train[Y_train_label_np == 1]
X_train_2 = X_train[Y_train_label_np == 2]
X_train_3 = X_train[Y_train_label_np == 3]
X_train_4 = X_train[Y_train_label_np == 4]
X_train_5 = X_train[Y_train_label_np == 5]
X_train_6 = X_train[Y_train_label_np == 6]
X_train_7 = X_train[Y_train_label_np == 7]
X_train_8 = X_train[Y_train_label_np == 8]
X_train_9 = X_train[Y_train_label_np == 9]
X_train_0 = X_train[Y_train_label_np == 0]


count=100
v1=np.vstack((X_train_0[:count],X_train_1[:count]))
v1=np.vstack((v1,X_train_2[:count]))
v1=np.vstack((v1,X_train_3[:count]))
v1=np.vstack((v1,X_train_4[:count]))
v1=np.vstack((v1,X_train_5[:count]))
v1=np.vstack((v1,X_train_6[:count]))
v1=np.vstack((v1,X_train_7[:count]))
v1=np.vstack((v1,X_train_8[:count]))
v1=np.vstack((v1,X_train_9[:count]))
v1_gen=np.empty_like(v1)
v2=np.zeros(count*10)
for x in range(count*10):
        v2[x]=x/count



print("Size of:")
print("- Training-set:\t\t{}".format(len(Y_train_lt5)))
print("- Test-set:\t\t{}".format(len(Y_train_lt5)))
print("- Validation-set:\t{}".format(len(Y_train_lt5)))



#data.test.cls = np.argmax(data.test.labels, axis=1)



# We know that MNIST images are 28 pixels in each dimension.
img_size = 28

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

# Number of classes, one class for each of 10 digits.
num_classes = 5

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
    


def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights




def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()
    
    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features




def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])



y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')


y_true_cls = tf.argmax(y_true, dimension=1)

layer_conv1, weights_conv1 = \
    new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)
                   
layer_conv2, weights_conv2 = \
    new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)
layer_flat, num_features = flatten_layer(layer_conv2)
layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)


layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)

y_pred = tf.nn.softmax(layer_fc2)


y_pred_cls = tf.argmax(y_pred, dimension=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,labels=y_true)


cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


session = tf.Session()

session.run(tf.global_variables_initializer())
train_batch_size = 1




# Counter for total number of iterations performed so far.
total_iterations = 0

def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        perm = np.arange(X_train_lt5.shape[0])
        np.random.shuffle(perm)
        perm=perm[:train_batch_size]
        x_batch = X_train_lt5[perm]
        y_true_batch = Y_train_lt5[perm]

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations.
        if i % 100 == 0:
            # Calculate the accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, acc))

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

def plot_images(content_image, mixed_image):
    # Create figure with sub-plots.
    fig, axes = plt.subplots(1, 2, figsize=(10, 10))

    # Adjust vertical spacing.
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    # Use interpolation to smooth pixels?
    smooth = True
    
    # Interpolation type.
    if smooth:
        interpolation = 'sinc'
    else:
        interpolation = 'nearest'

    # Plot the content-image.
    # Note that the pixel-values are normalized to
    # the [0.0, 1.0] range by dividing with 255.
    ax = axes.flat[0]
    content_image1=content_image.reshape((28,28))
    ax.imshow(content_image1, interpolation=interpolation)
    ax.set_xlabel("Content")

    # Plot the mixed-image.
    ax = axes.flat[1]
    mixed_image1=mixed_image.reshape((28,28))
    ax.imshow(mixed_image1, interpolation=interpolation)
    ax.set_xlabel("Mixed")


    # Remove ticks from all the plots.
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def plot_image_big(image):
    # Ensure the pixel-values are between 0 and 255.
    image = np.clip(image, 0.0, 1.0)
    image.reshape((28,28))
    # Convert pixels to bytes.
    image = image.astype(np.uint8)

    # Convert to a PIL-image and display it.
    display(PIL.Image.fromarray(image))

def mean_squared_error(a, b):
    return tf.reduce_mean(tf.square(a - b))

def create_content_loss(content_image):

    # Create a feed-dict with the content-image.
    feed_dict_train = {x: content_image}
    # Get references to the tensors for the given layers.

    # Calculate the output values of those layers when
    # feeding the content-image to the model.
    value1 = session.run(layer_conv1, feed_dict=feed_dict_train)
    value_const1=tf.constant(value1)
    loss_1=mean_squared_error(layer_conv1,value_const1)
    value2 = session.run(layer_conv2, feed_dict=feed_dict_train)
    value_const2=tf.constant(value2)
    loss_2=mean_squared_error(layer_conv2,value_const2)
    return (loss_1,loss_2)




def style_transfer(content_image,
                   num_iterations=200, step_size=0.01):
    content_image=content_image.reshape((1,784))
    (loss_content1,loss_content2) = create_content_loss(content_image=content_image)
    loss_combined = loss_content1 
 


    # Use TensorFlow to get the mathematical function for the
    # gradient of the combined loss-function with regard to
    # the input image.
    #gradient1 = tf.gradients(loss_content1, session.graph.get_tensor_by_name('x:0'))
    gradient2 = tf.gradients(loss_content1, session.graph.get_tensor_by_name('x:0'))
    # List of tensors that we will run in each optimization iteration.
    run_list = [gradient2]

    # The mixed-image is initialized with random noise.
    # It is the same size as the content-image.
    mixed_image = np.random.rand(*content_image.shape) + 128
    mixed_image=mixed_image.reshape((1,784))
    mixed_image=mixed_image/255.0
  
  #  mixed_image2 = np.random.rand(*content_image.shape) + 128
  #  mixed_image2=mixed_image2.reshape((1,784))
  #  mixed_image2=mixed_image2/255.0

    for i in range(num_iterations):
        # Create a feed-dict with the mixed-image.
        feed_dict_train = {x: mixed_image}

        # Use TensorFlow to calculate the value of the
        # gradient, as well as updating the adjustment values.
        grad = session.run(run_list, feed_dict=feed_dict_train)

        # Reduce the dimensionality of the gradient.
        grad = np.squeeze(grad)

        # Scale the step-size according to the gradient-values.
        step_size_scaled = step_size / (np.std(grad) + 1e-8)
        print "grad shape"
        print grad.shape
        print "mixed shape"
        print mixed_image.shape
        # Update the image by following the gradient.
        mixed_image -= grad* step_size_scaled

        # Ensure the image has valid pixel-values between 0 and 255.
        mixed_image = np.clip(mixed_image, 0.0, 1.0)

        # Print a little progress-indicator.
        print(". ")

        # Display status once every 10 iterations, and the last.
        if (i % 10 == 0) or (i == num_iterations - 1):
            print()
            print("Iteration:", i)
            # Plot the content-, style- and mixed-images.
            #plot_images(content_image=content_image,mixed_image=mixed_image)
            
    print()
    print("Final image:")
    #plot_image_big(mixed_image)

    # Close the TensorFlow session to release its resources.
    # Return the mixed-image.
    return mixed_image


saver=tf.train.Saver()
saver.save(session,'/home/rdx/personal/cog200/model_save/')

v1_gen=np.empty_like(v1)
def do_it():
    global v1_gen_layer2
    for i in range(v1.shape[0]):
        print "aaaaaaaaaaaaaaaaaaaaaaaaaa"
        print i
        print "bbbbbbbbbbbbbbbbbbbbbbbbbbb"
        ret=style_transfer(v1[i],step_size=0.001)
        ret=ret.reshape((784))
        v1_gen[i]=ret


np.savez('cog200_data_layer1_2.npz',
        v1=v1,
        v2=v2,
        v1_gen=v1_gen,
        v1_gen_layer2=v1_gen_layer2)

''''
In [64]: l1_sum_final.sum(axis=1)
Out[64]: 
array([ 8018.10351562,  5083.41992188,  7755.01660156,  7366.31933594,
        6782.44726562,  6271.58544922,  7448.06396484,  6603.953125  ,
        8119.54541016,  6839.11328125], dtype=float32)

In [65]: l2_sum_final.sum(axis=1)
Out[65]: 
array([ 10714.22363281,   7549.57910156,  10549.84765625,  10222.41210938,
         9637.2109375 ,   8856.26660156,  10270.82226562,   9235.93261719,
        10832.08398438,   9510.26074219], dtype=float32)



''''
'''''
without initial training
layer2
array([ 11414.13769531,   8240.32128906,  10980.546875  ,  10769.54296875,
        10000.38964844,   9862.296875  ,  10581.93261719,   9618.20898438,
        11293.93554688,  10074.81542969], dtype=float32)

        array([ 5004.32275391,  2900.2644043 ,  4790.91308594,  4503.68164062,
        3826.40258789,  3655.84521484,  4467.72851562,  3754.84179688,
        5071.10546875,  4040.72485352], dtype=float32)

''''