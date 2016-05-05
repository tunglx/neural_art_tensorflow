import tensorflow as tf
import numpy as np
from models import VGG16
import argparse
import scipy.misc
import os
from datetime import datetime as dt
import argparse
import time

# Add/Sub mean for image processing
mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)
def add_mean(img):
    for i in range(3):
        img[0,:,:,i] += mean[i]
    return img

def sub_mean(img):
    for i in range(3):
        img[0,:,:,i] -= mean[i]
    return img

# read image function
def read_image(path, w=None):
    img = scipy.misc.imread(path)
    # Resize if ratio is specified
    if w:
        r = w / np.float32(img.shape[1])
        img = scipy.misc.imresize(img, (int(img.shape[0]*r), int(img.shape[1]*r)))
    img = img.astype(np.float32)
    img = img[None, ...]
    # Subtract the image mean
    img = sub_mean(img)
    return img

# save image function
def save_image(im, iteration, out_dir):
    img = im.copy()
    # Add the image mean
    img = add_mean(img)
    img = np.clip(img[0, ...],0,255).astype(np.uint8)
    nowtime = dt.now().strftime('%Y_%m_%d_%H_%M_%S')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)    
    scipy.misc.imsave("{}/neural_art_{}_iteration{}.png".format(out_dir, nowtime, iteration), img)
   
# Make command to run program 
def parseArgs():
    parser = argparse.ArgumentParser(
        description='A Neural Algorithm of Artistic Style')
<<<<<<< HEAD
    parser.add_argument('--modelpath', '-p', default='VGG_model',
                        help='Model file path')
    parser.add_argument('--content', '-cp', default='images/samsung.jpg',
=======
    parser.add_argument('--model', '-m', default='vgg',
                        help='Model type (vgg, i2v, alexnet)')
    parser.add_argument('--modelpath', '-p', default='vgg',
                        help='Model file path')
    parser.add_argument('--content', '-cp', default='images/sd.jpg',
>>>>>>> 661989b2852200600703d64a89beb72d28d5dc23
                        help='Content image path')
    parser.add_argument('--style', '-sp', default='images/style.jpg',
                        help='Style image path')
    parser.add_argument('--iters', '-i', default=200, type=int,
                        help='Number of iterations')
    parser.add_argument('--alpha', '-a', default=1.0, type=float,
                        help='alpha (content weight)')
    parser.add_argument('--beta', '-b', default=200.0, type=float,
                        help='beta (style weight)')
    parser.add_argument('--device', default="/cpu:0")
    parser.add_argument('--out_dir', default="output")
    args = parser.parse_args()
<<<<<<< HEAD
    return args.content, args.style, args.modelpath, args.alpha, args.beta, args.iters, args.device, args

# Choose Convolutional model
def getModel(image, params_path):
    return VGG16(image, params_path)
=======
    return args.content, args.style, args.modelpath, args.model, args.alpha, args.beta, args.iters, args.device, args

# Choose Convolutional model
def getModel(image, params_path, model):
    if model == 'VGG':
        return VGG16(image, params_path)
    elif model == 'I2V':
        return I2V(image, params_path)
    elif model == 'ALEXNET':
        return Alexnet(image, params_path)
    else:
        print 'Invalid model name: use `VGG` or `I2V` or `ALEXNET`'
        return None

>>>>>>> 661989b2852200600703d64a89beb72d28d5dc23

# Main process

# Get value from run command 
<<<<<<< HEAD
content_image_path, style_image_path, params_path, alpha, beta, num_iters, device, args = parseArgs()
=======
content_image_path, style_image_path, params_path, modeltype, alpha, beta, num_iters, device, args = parseArgs()
>>>>>>> 661989b2852200600703d64a89beb72d28d5dc23
width = 600
# The actual calculation
print "Read images..."
content_image = read_image(content_image_path, width)
style_image   = read_image(style_image_path, width)
g = tf.Graph()
with g.device(device), g.as_default(), tf.Session(graph=g, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    print "Load content values..."
    image = tf.constant(content_image)
    model = getModel(image, params_path)
    content_image_y_val = [sess.run(y_l) for y_l in model.y()]  

    print "Load style values..."
    image = tf.constant(style_image)
    model = getModel(image, params_path)
    y = model.y()
    style_image_st_val = []
    for l in range(len(y)):
        num_filters = content_image_y_val[l].shape[3]
        st_shape = [-1, num_filters]
        st_ = tf.reshape(y[l], st_shape)
        st = tf.matmul(tf.transpose(st_), st_)
        style_image_st_val.append(sess.run(st)) 
    
    print "Construct graph..."
    # Start from white noise
    #gen_image = tf.Variable(tf.truncated_normal(content_image.shape, stddev=20), trainable=True, name='gen_image')
    # Start from the original image
    gen_image = tf.Variable(tf.constant(np.array(content_image, dtype=np.float32)), trainable=True, name='gen_image')
    model = getModel(gen_image, params_path)
    y = model.y()
    L_content = 0.0
    L_style   = 0.0
    for l in range(len(y)):
        # Content loss
        L_content += model.alpha[l]*tf.nn.l2_loss(y[l] - content_image_y_val[l])
        # Style loss
        num_filters = content_image_y_val[l].shape[3]
        st_shape = [-1, num_filters]
        st_ = tf.reshape(y[l], st_shape)
        st = tf.matmul(tf.transpose(st_), st_)
        N = np.prod(content_image_y_val[l].shape).astype(np.float32)
        L_style += model.beta[l]*tf.nn.l2_loss(st - style_image_st_val[l])/N**2/len(y)
    # The loss
    L = alpha* L_content + beta * L_style
    # The optimizer
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=2.0, global_step=global_step, decay_steps=100, decay_rate=0.94, staircase=True)
    train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(L, global_step=global_step)
    
    # Set up the summary writer (saving summaries is optional)
    tf.scalar_summary("L_content", L_content)
    tf.scalar_summary("L_style", L_style)
    gen_image_addmean = tf.Variable(tf.constant(np.array(content_image, dtype=np.float32)), trainable=False)
    tf.image_summary("Generated image (TODO: add mean)", gen_image_addmean)
    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter('/tmp/na-logs', graph_def=sess.graph_def)
    
    print "Start calculation..."
    # The optimizer has variables that require initialization as well
    sess.run(tf.initialize_all_variables())
    for i in range(num_iters):
        if i % 10 == 0:
            gen_image_val = sess.run(gen_image)
            save_image(gen_image_val, i, args.out_dir)
            print "L_content: ", sess.run(L_content), "L_style: ", sess.run(L_style)
            # Increment summary
            sess.run(tf.assign(gen_image_addmean, add_mean(gen_image_val)))
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, i)
        print "Iter:", i
        sess.run(train_step)

