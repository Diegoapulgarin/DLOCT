#%%
import tensorflow as tf
from tensorflow import keras
# import tensorflow_addons as tfa


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pathlib
import time
import datetime
import numpy as np
import scipy.io as sio

from matplotlib import pyplot as plt
from IPython import display
#%%
# path to training data pairs
train_path = "../train"
# path to testing data pairs
test_path = "../test"
# path to save log files
log_path = "../logs" 
# path to save check point data files
ckpt_path = "../ckpt"

BATCH_SIZE = 1
# Each image is 1024 x 1024 in size
IMG_WIDTH = 1024
IMG_HEIGHT = 1024
#%%
# Loads data and changes the constrast dynamically for training 

def read_mat(image_path):
    # Input mat files are 10*log10(tomogram intensity)
    image = sio.loadmat(image_path.numpy())['bscanStack']
    # Computes the dynamic contrast range 
    noiseFloordB = np.round(np.mean(image[10:20,200:250,:]))
    noiseFloordB = noiseFloordB
    if np.random.uniform() > 0.5:
        noiseFloordB = noiseFloordB + np.round(np.random.uniform(-1,10))
        
    try:
        ind = np.unravel_index(np.argmax(image, axis = None), image.shape)
        maximumdB = np.round(np.mean(image[ind(0)-5:ind(0)+5, ind(1)-5:ind(1)+5, ind(2)-1:ind(2)+1]))-10   
    except:
        maximumdB = np.max(image[:,:,17:])-10
    if np.random.uniform() > 0.5:
        maximumdB = maximumdB + np.round(np.random.uniform(-15,1))
    
    # Quantizes the intensity data into a 16-bit values                          
    image = (image-noiseFloordB)/(maximumdB-noiseFloordB)
    image = image*(2**16-1)
    image = np.clip(image, 0, 2**16-1)
    data = image.astype(np.uint16) 
    return data

def read_mat2(image_path):
    # Input mat files are 10*log10(tomogram intensity)
    try:
        image = sio.loadmat(image_path.numpy())['bscanStack']
    except:
        image = sio.loadmat(image_path.numpy())['image_stack']
        
    #noiseFloordB = np.round(np.mean(image[1:20,200:250,:]))
#     noiseFloordB = noiseFloordB+2.50
    noiseFloordB = 50;
    maximumdB = 250;
    # Compute the noise Floor
#     try:
#        ind = np.unravel_index(np.argmax(image, axis=None), image.shape)
#        maximumdB = np.round(np.mean(image[ind(0)-5:ind(0)+5, ind(1)-5:ind(1)+5, ind(2)-2:ind(2)+2]))   
#     except:
#        maximumdB = np.max(image[:,:,8])
                             
    image = (image-noiseFloordB)/(maximumdB-noiseFloordB)
    image = image*(2**16-1)
    image = np.clip(image, 0, 2**16-1)
    data = image.astype(np.uint16) 
    return data

def load_train(image_path):
  image = tf.py_function(func=read_mat, inp = [image_path], Tout = tf.uint16)
  image = tf.image.convert_image_dtype(image, dtype = tf.uint16, saturate = False)

  maxImageDimension = tf.math.reduce_max(tf.shape(image))
  image = tf.image.resize_with_crop_or_pad(image, maxImageDimension, maxImageDimension)
  
  # Converts both images to float32 tensors
  input_image_stack = tf.cast(image, tf.float32)
  #real_image = tf.cast(real_image, tf.float32)
  return input_image_stack

def load_test(image_path):
  image = tf.py_function(func=read_mat2, inp=[image_path], Tout=tf.uint16)
  #image = tf.convert_to_tensor(image, dtype=tf.uint8)
  image = tf.image.convert_image_dtype(image, dtype=tf.uint16, saturate=False)
  maxImageDimension = tf.math.reduce_max(tf.shape(image))
  image = tf.image.resize_with_crop_or_pad(image, maxImageDimension, maxImageDimension)
    # Convert both images to float32 tensors
  input_image_stack = tf.cast(image, tf.float32)
  #real_image = tf.cast(real_image, tf.float32)
  return input_image_stack

def resize(input_image_stack, height, width):
  input_image_stack = tf.image.resize_with_crop_or_pad(input_image_stack, height, width)
  #real_image = tf.image.resize(real_image, [height, width],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  return input_image_stack

# Normalizing the images to [-1, 1]
def normalize(input_image_stack):
  input_image_stack = (input_image_stack/(0.5*65535))-1
  #input_image_stack = (input_image_stack / (0.5*tf.math.reduce_max(input_image_stack))) - 1
  #real_image = (real_image / 127.5) - 1
  return input_image_stack

def random_crop(input_image_stack):
  cropped_image = tf.image.random_crop(input_image_stack, size=[IMG_HEIGHT, IMG_WIDTH, tf.shape(input_image_stack)[2]])
  return cropped_image

@tf.function()
def augment_data(input_image_stack):
  # Resizing to 1144 x 1144
  input_image_stack = resize(input_image_stack, 1144, 1144)

  # Random cropping back to 1024 x 1024
  input_image_stack = random_crop(input_image_stack)

  if tf.random.uniform(()) > 0.50:

    # Random mirroring
    input_image_stack = tf.image.flip_left_right(input_image_stack)
    
    # Random Rotation of the image with angles minval: maxval
    deg = tf.math.round(tf.random.uniform(shape = [1], minval = -10, maxval = 10))
    deg = np.pi*deg/180
    input_image_stack = tfa.image.rotate(input_image_stack, deg)
    
    # Random x,y translation
    trans_xy = tf.random.uniform(shape = [1,2], minval = -50, maxval = 50, dtype = tf.dtypes.float32)                        
    input_image_stack = tfa.image.translate(input_image_stack, trans_xy, interpolation = 'nearest', fill_mode = 'constant', fill_value = 0.0)
     
  return input_image_stack

def load_image_train(image_file):
  input_image_stack = load_train(image_file)
  input_image_stack = augment_data(input_image_stack)
  input_image_stack = normalize(input_image_stack)
  return input_image_stack[:,:,:17], input_image_stack[:,:,17:]

def load_image_test(image_file):
  input_image_stack = load_test(image_file)
  input_image_stack = resize(input_image_stack, IMG_HEIGHT, IMG_WIDTH)
  input_image_stack = normalize(input_image_stack)
  return input_image_stack[:,:,:17], input_image_stack[:,:,17:]

def load_image_test_new(image_file):
  input_image, real_image = load_test(image_file)
  input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)
  #return input_image, real_image
  return input_image, real_image, image_file
# The batch size of 1 produced better results for the U-Net in the original pix2pix experiment

def downsample(filters, size, apply_batchnorm = True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides = 2, padding ='same',
                             kernel_initializer = initializer, use_bias = True))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

def upsample(filters, size, apply_dropout = False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides = 2,
                                    padding = 'same',
                                    kernel_initializer = initializer,
                                    use_bias = True))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

def Generator():
  inputs = tf.keras.layers.Input(shape=[1024, 1024, 17])

  down_stack = [
    downsample(256, 4, apply_batchnorm = False),  # (batch_size, 512, 512, 256)
    downsample(512, 4),   # (batch_size, 256, 256, 512)
    downsample(1024, 4),  # (batch_size, 128, 128, 1024)
    downsample(2048, 4),  # (batch_size, 64, 64, 2048)
    downsample(2048, 4),  # (batch_size, 32, 32, 2048)
  ]

  up_stack = [
    upsample(2048, 4),  # (batch_size, 64, 64, 2048)
    upsample(1024, 4),  # (batch_size, 128, 128, 1024)
    upsample(512, 4),   # (batch_size, 256, 256, 512)
    upsample(256, 4),   # (batch_size, 512, 512, 256)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 1024, 1024, 1)
  x = inputs
  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # Mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
  total_gen_loss = gan_loss + (LAMBDA * l1_loss)
  return total_gen_loss, gan_loss, l1_loss

def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  input = tf.keras.layers.Input(shape=[1024, 1024, 17], name = 'input_image')
  target = tf.keras.layers.Input(shape=[1024, 1024, 1], name = 'target_image')

  x = tf.keras.layers.concatenate([input, target])  # (batch_size, 1024, 1024, channels*2)

  down1 = downsample(512, 4, False)(x)  # (batch_size, 512, 512, 64)
  down2 = downsample(1024, 4)(down1)  # (batch_size, 256, 256, 128)
  down3 = downsample(1024, 4)(down2)  # (batch_size, 128, 128, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 130, 130, 256)
  conv = tf.keras.layers.Conv2D(2048, 4, strides = 1,
                                kernel_initializer = initializer,
                                use_bias = False)(zero_pad1)  # (batch_size, 127, 127, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 129, 129, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides = 1,
                                kernel_initializer = initializer)(zero_pad2)  # (batch_size, 126, 126, 1)

  return tf.keras.Model(inputs=[input, target], outputs = last)

def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
  total_disc_loss = real_loss + generated_loss
  return total_disc_loss, real_loss, generated_loss

def generate_images(model, test_input, tar, filename):
  start =  time.time()
  prediction = model(test_input, training = True)
  print(f'Time taken: {time.time()-start:.2f} sec\n')
  panel = tf.concat([tf.squeeze(test_input[0][:,:,8]), tf.squeeze(tar[0]), tf.squeeze(prediction[0])],1)
  panel = tf.expand_dims(panel,2)
  encoded = tf.io.encode_png(tf.cast((panel+1)*127.5, dtype = 'uint8'))
  outputfile = tf.io.write_file(filename, encoded)
  plt.figure(figsize=(15, 15))
  display_list = [test_input[0][:,:,8], tar[0], prediction[0]]
  mae = tf.keras.losses.MeanAbsoluteError()
  print(f'L1_error:{mae(tar[0], prediction[0]).numpy()}')
  
  title = ['Input Image', 'Ground Truth', 'Predicted Image']
  
  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # Getting the pixel values in the [0, 1] range to plot.
    plt.imshow(tf.squeeze(display_list[i]) * 0.5 + 0.5, cmap ='gray')
    plt.axis('off')   
  plt.show()

@tf.function
def train_step(input_image, target, step):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training = True)

    disc_real_output = discriminator([input_image, target], training = True)
    disc_generated_output = discriminator([input_image, gen_output], training = True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss, real_loss, fake_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step = step//BUFFER_SIZE)
    tf.summary.scalar('real_loss', real_loss, step = step//BUFFER_SIZE)
    tf.summary.scalar('fake_loss', fake_loss, step = step//BUFFER_SIZE)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step = step//BUFFER_SIZE)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step = step//BUFFER_SIZE)
    tf.summary.scalar('disc_loss', disc_loss, step = step//BUFFER_SIZE)

def fit(train_ds, test_ds, steps):
  example_input, example_target = next(iter(test_ds.take(1)))
  start = time.time()

  for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
    if (step) % BUFFER_SIZE == 0:
      display.clear_output(wait = True)

      if step != 0:
        print(f'Time taken for last Epoch: {time.time()-start:.2f} sec\n')
        
      start = time.time()
      generate_images(generator, example_input, example_target)
      print(f"Epoch: {step//BUFFER_SIZE}")

    train_step(input_image, target, step)

    # Training step
    if (step+1) % 10 == 0:
      print('.', end='', flush = True)

    # Save (checkpoint) the model every 10 epochs
    if (step + 1) % (BUFFER_SIZE*10) == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)
#%%

train_dataset = tf.data.Dataset.list_files(train_path + "/*.mat")
BUFFER_SIZE = train_dataset.cardinality().numpy()
print(BUFFER_SIZE)
train_dataset = train_dataset.map(load_image_train, num_parallel_calls = tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

try:
  test_dataset = tf.data.Dataset.list_files(test_path + "/*.mat")
except tf.errors.InvalidArgumentError:
  test_dataset = tf.data.Dataset.list_files(test_path + "/*.mat")
print(test_dataset.cardinality().numpy())
test_dataset = test_dataset.map(load_image_train)
test_dataset = test_dataset.batch(BATCH_SIZE)

OUTPUT_CHANNELS = 1

generator = Generator()
tf.keras.utils.plot_model(generator, show_shapes = True, dpi = 64)
gen_output = generator(inputs[tf.newaxis, ...], training = False)
plt.imshow(tf.squeeze(gen_output[0, ...]))

LAMBDA = 1000
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits = True)

discriminator = Discriminator()
tf.keras.utils.plot_model(discriminator, show_shapes = True, dpi = 64)

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1 = 0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1 = 0.5)

checkpoint_dir = "/data/PersonalFolders/bhaskara/" + ckpt_path
checkpoint_prefix = os.path.join(checkpoint_dir, "")
print(checkpoint_prefix)

checkpoint = tf.train.Checkpoint(generator_optimizer = generator_optimizer,
                                 discriminator_optimizer = discriminator_optimizer,
                                 generator = generator,
                                 discriminator = discriminator)

for example_input, example_target in test_dataset.take(1):
  generate_images(generator, example_input, example_target)

summary_writer = tf.summary.create_file_writer(
  log_path + "/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

print(BUFFER_SIZE)

# #/data/PersonalFolders/bhaskara/despeckle/logs
# print(log_path)
# log_dir = "/data/PersonalFolders/bhaskara/" + log_path
# %reload_ext tensorboard
# %tensorboard --logdir "/data/PersonalFolders/bhaskara/despeckle/Experiments_Spectralis/logs"

fit(train_dataset, test_dataset, steps = BUFFER_SIZE*200)

for input, target in test_dataset.take(5):
  generate_images(generator, input, target)
