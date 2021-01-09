
# coding: utf-8

# In[1]:


# from google.colab import drive
# drive.mount('/content/drive/')


# In[2]:


# from tensorflow.python.client import device_lib
# device_lib.list_local_devices()


# In[1]:


import os
import pickle
import random
import time


# In[2]:


import PIL
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from keras import Input, Model
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.layers import Dense, LeakyReLU, BatchNormalization, ReLU, Reshape, UpSampling2D, Conv2D, Activation,     concatenate, Flatten, Lambda, Concatenate
from keras.optimizers import Adam
from keras.layers import *
from matplotlib import pyplot as plt


# In[3]:





dir_to_change = 'C:/Users/riagu/Documents/'



# In[4]:


# test_file=open("/content/drive/My Drive/major proj 2/birds/example_captions.txt",'r')
# test_file.readlines()


# In[5]:


# with open("/content/drive/My Drive/major proj 2/birds/train/char-CNN-RNN-embeddings.pickle",'rb') as f:
#   embeddings=pickle.load(f,encoding='latin1')
#   embeddings=np.array(embeddings)
#   print('embeddings: ',embeddings.shape)


# ### Loading Dataset
# 

# In[10]:


def load_class_ids(class_info_file_path):
    with open(class_info_file_path,'rb') as f:
        class_ids=pickle.load(f,encoding='latin1')
    return class_ids


# In[12]:


def load_embeddings(embeddings_file_path):
    with open(embeddings_file_path,'rb')as f:
        embeddings=pickle.load(f,encoding='latin1')
        embeddings=np.array(embeddings)
#    print('embeddings: ',embeddings.shape)
    return embeddings  


# In[13]:


def load_filenames(filenames_file_path):
    with open(filenames_file_path,'rb') as f:
        filenames = pickle.load(f, encoding='latin1')
    return filenames  


# In[14]:


def load_bounding_boxes(dataset_dir):
    # Paths
    bounding_boxes_path = os.path.join(dataset_dir, 'bounding_boxes.txt')
    file_paths_path = os.path.join(dataset_dir, 'images.txt')

    # Read bounding_boxes.txt and images.txt file
    df_bounding_boxes = pd.read_csv(bounding_boxes_path,
                                    delim_whitespace=True, header=None).astype(int)
    df_file_names = pd.read_csv(file_paths_path, delim_whitespace=True, header=None)

    # Create a list of file names
    file_names = df_file_names[1].tolist()

    # Create a dictionary of file_names and bounding boxes
    filename_boundingbox_dict = {img_file[:-4]: [] for img_file in file_names[:2]}

    # Assign a bounding box to the corresponding image
    for i in range(0, len(file_names)):
        # Get the bounding box
        bounding_box = df_bounding_boxes.iloc[i][1:].tolist()
        key = file_names[i][:-4]
        filename_boundingbox_dict[key] = bounding_box

    return filename_boundingbox_dict


# In[15]:


def get_img(img_path, bbox, image_size):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        R = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - R)
        y2 = np.minimum(height, center_y + R)
        x1 = np.maximum(0, center_x - R)
        x2 = np.minimum(width, center_x + R)
        img = img.crop([x1, y1, x2, y2])
    img = img.resize(image_size, PIL.Image.BILINEAR)
    return img


# In[16]:


def load_dataset(filenames_file_path, class_info_file_path, cub_dataset_dir, embeddings_file_path, image_size):
    filenames = load_filenames(filenames_file_path)
    class_ids = load_class_ids(class_info_file_path)
    bounding_boxes = load_bounding_boxes(cub_dataset_dir)
    all_embeddings = load_embeddings(embeddings_file_path)

    X, y, embeddings = [], [], []

#    print("Embeddings shape:", all_embeddings.shape)

    for index, filename in enumerate(filenames):
        bounding_box = bounding_boxes[filename]

        try:
            # Load images
            img_name = '{}/images/{}.jpg'.format(cub_dataset_dir, filename)
            img = get_img(img_name, bounding_box, image_size)

            all_embeddings1 = all_embeddings[index, :, :]

            embedding_ix = random.randint(0, all_embeddings1.shape[0] - 1)
            embedding = all_embeddings1[embedding_ix, :]

            X.append(np.array(img))
            y.append(class_ids[index])
            embeddings.append(embedding)
        except Exception as e:
#            print(e)

    X = np.array(X)
    y = np.array(y)
    embeddings = np.array(embeddings)
    
    X = X[:5120]
    y = y[:5120]
    embeddings = embeddings[:5120]
    return X, y, embeddings


# ### Model Creation

# In[17]:


def generate_c(x):
    mean=x[:,:128]
    log_sigma = x[:, 128:]
    stddev=K.exp(log_sigma)
    epsilon=K.random_normal(shape=K.constant((mean.shape[1],), dtype='int32'))
    c=stddev*epsilon + mean
    return c
  


# In[18]:


def build_ca_model():
    input_layer=Input(shape=(1024,))
    x=Dense(256)(input_layer)
    x=LeakyReLU(alpha=0.2)(x)
    model=Model(inputs=[input_layer],outputs=[x])
    return model


# In[19]:


def build_embedding_compressor_model():
    input_layer=Input(shape=(1024,))
    x=Dense(128)(input_layer)
    x=ReLU()(x)
    model=Model(inputs=[input_layer],outputs=[x])
    return model


# In[20]:


def build_stage1_generator():
    input_layer = Input(shape=(1024,))
    x = Dense(256)(input_layer)
    mean_logsigma = LeakyReLU(alpha=0.2)(x)

    c = Lambda(generate_c)(mean_logsigma)

    input_layer2 = Input(shape=(100,))

    gen_input = Concatenate(axis=1)([c, input_layer2])

    x = Dense(128 * 8 * 4 * 4, use_bias=False)(gen_input)
    x = ReLU()(x)

    x = Reshape((4, 4, 128 * 8), input_shape=(128 * 8 * 4 * 4,))(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(512, kernel_size=3, padding="same", strides=1, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(256, kernel_size=3, padding="same", strides=1, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(128, kernel_size=3, padding="same", strides=1, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, kernel_size=3, padding="same", strides=1, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(3, kernel_size=3, padding="same", strides=1, use_bias=False)(x)
    x = Activation(activation='tanh')(x)

    stage1_gen = Model(inputs=[input_layer, input_layer2], outputs=[x, mean_logsigma])
    return stage1_gen
  


# In[21]:


def build_stage1_discriminator():
    input_layer = Input(shape=(64, 64, 3))

    x = Conv2D(64, (4, 4),
               padding='same', strides=2,
               input_shape=(64, 64, 3), use_bias=False)(input_layer)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(128, (4, 4), padding='same', strides=2, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(256, (4, 4), padding='same', strides=2, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(512, (4, 4), padding='same', strides=2, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    input_layer2 = Input(shape=(4, 4, 128))

    merged_input = concatenate([x, input_layer2])

    x2 = Conv2D(64 * 8, kernel_size=1,
                padding="same", strides=1)(merged_input)
    x2 = BatchNormalization()(x2)
    x2 = LeakyReLU(alpha=0.2)(x2)
    x2 = Flatten()(x2)
    x2 = Dense(1)(x2)
    #x2 = Activation('sigmoid')(x2)

    stage1_dis = Model(inputs=[input_layer, input_layer2], outputs=[x2])
    return stage1_dis


# In[22]:


def build_adversarial_model(gen_model, dis_model):
    input_layer = Input(shape=(1024,))
    input_layer2 = Input(shape=(100,))
    input_layer3 = Input(shape=(4, 4, 128))

    x, mean_logsigma = gen_model([input_layer, input_layer2])

    dis_model.trainable = False
    valid = dis_model([x, input_layer3])

    model = Model(inputs=[input_layer, input_layer2, input_layer3], outputs=[valid, mean_logsigma])
    return model


# ### Loss functions

# In[23]:


def KL_loss(y_true, y_pred):
    mean = y_pred[:, :128]
    logsigma = y_pred[:, :128]
    loss = -logsigma + .5 * (-1 + K.exp(2. * logsigma) + K.square(mean))
    loss = K.mean(loss)
    return loss


# In[24]:


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


# In[25]:


def custom_generator_loss(y_true, y_pred):
    # Calculate binary cross entropy loss
    return K.binary_crossentropy(y_true, y_pred)


# In[26]:


def save_rgb_img(img, path):
    """
    Save an rgb image
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("Image")

    plt.savefig(path)
    plt.close()


# In[27]:


def write_log(callback, name, loss, batch_no):
    """
    Write training summary to TensorBoard
    """
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = loss
    summary_value.tag = name
    callback.writer.add_summary(summary, batch_no)
    callback.writer.flush()


# ### Downloading and extracting dataset

# In[28]:


# !wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz


# In[29]:


# import tarfile
# tar = tarfile.open("CUB_200_2011.tgz")
# tar.extractall()
# tar.close()


# In[30]:


# data_dir = "/content/drive/My Drive/major proj 2/birds/"
train_dir = dir_to_change + "birds/train"
test_dir = dir_to_change + "birds/test"
image_size = 64
batch_size = 64
z_dim = 100
stage1_generator_lr = 0.0002
stage1_discriminator_lr = 0.0002
stage1_lr_decay_step = 600
epochs = 80000
condition_dim = 128

embeddings_file_path_train = train_dir + "/char-CNN-RNN-embeddings.pickle"
embeddings_file_path_test = test_dir + "/char-CNN-RNN-embeddings.pickle"

filenames_file_path_train = train_dir + "/filenames.pickle"
filenames_file_path_test = test_dir + "/filenames.pickle"

class_info_file_path_train = train_dir + "/class_info.pickle"
class_info_file_path_test = test_dir + "/class_info.pickle"

cub_dataset_dir = dir_to_change + "CUB_200_2011/CUB_200_2011"
    


# In[31]:


dis_optimizer = Adam(lr=stage1_discriminator_lr, beta_1=0.5, beta_2=0.999)
gen_optimizer = Adam(lr=stage1_generator_lr, beta_1=0.5, beta_2=0.999)


# In[32]:


X_train, y_train, embeddings_train = load_dataset(filenames_file_path=filenames_file_path_train,
                                                      class_info_file_path=class_info_file_path_train,
                                                      cub_dataset_dir=cub_dataset_dir,
                                                      embeddings_file_path=embeddings_file_path_train,
                                                      image_size=(64, 64))

X_test, y_test, embeddings_test = load_dataset(filenames_file_path=filenames_file_path_test,
                                                   class_info_file_path=class_info_file_path_test,
                                                   cub_dataset_dir=cub_dataset_dir,
                                                   embeddings_file_path=embeddings_file_path_test,
                                                   image_size=(64, 64))


# In[33]:


#print(X_train.shape)
#print(y_train.shape)
#print(embeddings_train.shape)
#print(X_test.shape)
#print(y_test.shape)
#print(embeddings_test.shape)


# ### Building and Compiling model

# In[34]:


from keras.optimizers import *
optimizer = RMSprop(lr=0.00005)

ca_model = build_ca_model()
ca_model.compile(loss="binary_crossentropy", optimizer="adam",metrics=['accuracy'])

stage1_dis = build_stage1_discriminator()
stage1_dis.compile(loss=wasserstein_loss, optimizer=optimizer,metrics=['accuracy'])

stage1_gen = build_stage1_generator()
stage1_gen.compile(loss=wasserstein_loss,optimizer=optimizer,metrics=['accuracy'])

embedding_compressor_model = build_embedding_compressor_model()
embedding_compressor_model.compile(loss="binary_crossentropy", optimizer="adam",metrics=['accuracy'])

adversarial_model = build_adversarial_model(gen_model=stage1_gen, dis_model=stage1_dis)
adversarial_model.compile(loss=wasserstein_loss,
                              optimizer=optimizer, metrics=['accuracy'])

#edited log folder name
log_folder = dir_to_change + "Output_Folder/stage1_wgan_logs/"
tensorboard = TensorBoard(log_dir=log_folder.format(time.time()),write_graph=True,batch_size=64,update_freq='epoch')
tensorboard.set_model(stage1_gen)
tensorboard.set_model(stage1_dis)
tensorboard.set_model(ca_model)
tensorboard.set_model(embedding_compressor_model)


# In[35]:


# Generate an array containing real and fake values
real_labels = -np.ones((64, 1))
fake_labels = np.ones((64, 1))
#wrong_labels = np.zeros((batch_size, 1), dtype=float) * 0.1


# In[37]:


d_loss_per_epoch=[]
g_loss_per_epoch=[]
# epochs = 9
# print(epochs)
n_critic = 8
dir_results =dir_to_change +"Output_Folder/stage1_gan_results/"
for epoch in range(epochs):
#        print("========================================")
#        print("Epoch is:", epoch)
        #print("Number of batches", int(X_train.shape[0] / batch_size))

        gen_losses = []
        dis_losses = []
        
        for index in range(n_critic):
          
          # ---------------------
          #  Train Discriminator
          # ---------------------
          
          # Sample a batch of data
            z_noise = np.random.normal(0, 1, size=(batch_size, z_dim))
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            #image_batch = X_train[index *batch_size:(index + 1) * batch_size]
            image_batch = X_train[idx]
            #embedding_batch = embeddings_train[index * batch_size:(index + 1) *batch_size]
            embedding_batch = embeddings_train[idx]
            image_batch = (image_batch - 127.5) / 127.5
            
          # Generate fake images
            fake_images, _ = stage1_gen.predict([embedding_batch, z_noise], verbose=3)

          # Generate compressed embeddings
            compressed_embedding = embedding_compressor_model.predict_on_batch(embedding_batch)
            compressed_embedding = np.reshape(compressed_embedding, (-1, 1, 1, condition_dim))
            compressed_embedding = np.tile(compressed_embedding, (1, 4, 4, 1))
          
          # Train the critic
            dis_loss_real = stage1_dis.train_on_batch([image_batch, compressed_embedding],
                                                       real_labels)
            
            dis_loss_fake = stage1_dis.train_on_batch([fake_images, compressed_embedding],
                                                      fake_labels)
            
            d_loss = 0.5 * np.add(dis_loss_fake, dis_loss_real)
            
            #dis_losses.append(d_loss)
            
            for l in stage1_dis.layers:
                weights= l.get_weights()
                weights = [np.clip(w, -0.01, 0.01) for w in weights]
                l.set_weights(weights)
        
        # ---------------------
        #  Train Generator
        # ---------------------
        
#        print("Training the combined generator and critic")
        g_loss = adversarial_model.train_on_batch([embedding_batch, z_noise, compressed_embedding],[K.ones((batch_size, 1)) * -1, K.ones((batch_size, 256)) * 1])
        
#        print("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))
        
        write_log(tensorboard, 'discriminator_loss_wgan',1 - d_loss[0] , epoch)
        write_log(tensorboard, 'generator_loss_wgan',1 - g_loss[0] , epoch)
        
        d_loss_per_epoch.append(1 - d_loss[0])
        g_loss_per_epoch.append(1 - g_loss[0])
        
        # Generate and save images after every 2nd epoch
        if epoch % 2 == 0:
            # z_noise2 = np.random.uniform(-1, 1, size=(batch_size, z_dim))
            z_noise2 = np.random.normal(0, 1, size=(batch_size, z_dim))
            embedding_batch = embeddings_test[0:batch_size]
            fake_images, _ = stage1_gen.predict_on_batch([embedding_batch, z_noise2])

            # Save images # edited folder name stage 1 wgan results
            for i, img in enumerate(fake_images[:10]):
                save_rgb_img(img, (dir_results+"gen_{}_{}.jpg".format(epoch, i)))

        dir_weights = dir_to_change+"Output_Folder/"
        
        # Save models
        stage1_gen.save_weights(dir_weights+"stage1_gen_wgan.h5")
        stage1_dis.save_weights(dir_weights+"stage1_dis_wgan.h5")


# In[38]:


epoch_count = range(1, len(d_loss_per_epoch) + 1)
plt.plot(epoch_count, d_loss_per_epoch, 'r', label="Discriminator Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
# plt.legend(['Discriminator Loss'])
plt.legend()
plt.title("Stage 1: Discriminator Loss")
# plt.show()
dir_plot = dir_to_change + "Output_Folder/"
plt.savefig(dir_plot+"Stage1_DiscriminatorLoss")


# In[39]:


epoch_count = range(1, len(d_loss_per_epoch) + 1)
plt.plot(epoch_count, g_loss_per_epoch, 'b', label="Generator Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
# plt.legend(['Generator Loss'])
plt.legend()
plt.title("Stage 1: Generator Loss")
# plt.show()
dir_plot = dir_to_change + "Output_Folder/"
plt.savefig(dir_plot+"Stage1_GeneratorLoss")

df = pd.DataFrame(data={"col1": d_loss_per_epoch, "col2": g_loss_per_epoch})
df.to_csv("./Model 2 stage1 losses.csv", sep=',',index=False)


# In[ ]:


# !pip install -U -q PyDrive
# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive
# from google.colab import auth
# from oauth2client.client import GoogleCredentials


# In[ ]:


# auth.authenticate_user()
# gauth = GoogleAuth()
# gauth.credentials = GoogleCredentials.get_application_default()
# drive = GoogleDrive(gauth)


# In[ ]:


# uploaded = drive.CreateFile({'title': 'stage1_dis_wgan.h5'})
# uploaded.SetContentFile('stage1_dis_wgan.h5')
# uploaded.Upload()
# print('Uploaded file with ID {}'.format(uploaded.get('id')))


# In[ ]:


# uploaded = drive.CreateFile({'title': 'stage1_gen_wgan.h5'})
# uploaded.SetContentFile('stage1_gen_wgan.h5')
# uploaded.Upload()
# print('Uploaded file with ID {}'.format(uploaded.get('id')))

