import tempfile
import os
import pickle
import random as random_state
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from base_provider import ImagesDataSet, DataProvider
#%matplotlib inline
def augment_image(image, pad):
    """Perform zero padding, randomly crop image to original size,
    maybe mirror horizontally"""
    #flip = random.getrandbits(1)
    flip = random_state.uniform(0,1)
    #if flip>0.5:
    #    image = image[:, ::-1, :]
    
    init_shape = image.shape
    new_shape = [init_shape[0] + pad * 2,
                 init_shape[1] + pad * 2,
                 init_shape[2]]
    zeros_padded = np.zeros(new_shape)
    zeros_padded[pad:init_shape[0] + pad, pad:init_shape[1] + pad, :] = image
    # randomly crop to original size
    init_x = np.random.randint(0, pad * 2)
    init_y = np.random.randint(0, pad * 2)
    cropped = zeros_padded[
        init_x: init_x + init_shape[0],
        init_y: init_y + init_shape[1],
        :]
    flip = random_state.getrandbits(1)
    if flip:
       cropped = cropped[:, ::-1, :]
    return image


def augment_all_images(initial_images, pad):
    new_images = np.zeros(initial_images.shape)
    for i in range(initial_images.shape[0]):
        new_images[i] = augment_image(initial_images[i], pad=4)
    return new_images

class CamvidDataSet(ImagesDataSet):
    def __init__(self, images, labels, n_classes, shuffle, normalization,
                 augmentation):
        """
        Args:
            images: 4D numpy array
            labels: 2D or 1D numpy array
            n_classes: `int`, number of cifar classes - 10 or 100
            shuffle: `str` or None
                None: no any shuffling
                once_prior_train: shuffle train data only once prior train
                every_epoch: shuffle train data prior every epoch
            normalization: `str` or None
                None: no any normalization
                divide_255: divide all pixels by 255
                divide_256: divide all pixels by 256
                by_chanels: substract mean of every chanel and divide each
                    chanel data by it's standart deviation
            augmentation: `bool`
        """
        if shuffle is None:
            self.shuffle_every_epoch = False
        elif shuffle == 'once_prior_train':
            self.shuffle_every_epoch = False
            images, labels = self.shuffle_images_and_labels(images, labels)
        elif shuffle == 'every_epoch':
        	  print("Apply Shuffle")
            self.shuffle_every_epoch = True
        else:
            raise Exception("Unknown type of shuffling")

        self.images = images
        self.labels = labels
        self.n_classes = n_classes
        self.augmentation = augmentation
        self.normalization = normalization

        if normalization:
        	  print("Apply Normalization")
            #print("augementation is %s",self.augmentation)
            self.images = self.normalize_images(images, self.normalization)
            self.start_new_epoch()

    def start_new_epoch(self):
        self._batch_counter = 0
        if self.shuffle_every_epoch:
            images, labels = self.shuffle_images_and_labels(
                self.images, self.labels)
        else:
            images, labels = self.images, self.labels
        if self.augmentation:
        	  print("Apply Augmentation")
            images = augment_all_images(images, pad=4)
        #images = self.normalize_images(images, self.normalization)
        self.epoch_images = images
        self.epoch_labels = labels
        #self.epoch_images = images[0:5000]
        #self.epoch_labels = labels[0:5000]

    @property
    def num_examples(self):
        return self.labels.shape[0]

    def next_batch(self, batch_size):
        start = self._batch_counter * batch_size
        end = (self._batch_counter + 1) * batch_size
        self._batch_counter += 1
        images_slice = self.epoch_images[start: end]
        labels_slice = self.epoch_labels[start: end]
        if images_slice.shape[0] != batch_size:
            self.start_new_epoch()
            return self.next_batch(batch_size)
        else:
            #print('labels_slice'+str(labels_slice.shape))
            test = self.one_hot_it(labels_slice,224,224)
            #print('test'+str(test.shape))
            #result = np.zeros_like(test)
            #for i in range(0,11):
            #    result[:,:,:,i] = cv2.medianBlur(test[:,:,:,i].astype(np.uint8),5)
            return images_slice, test
    
    def one_hot_it(self, labels,w,h):
        # print("labels in one hot"+str(labels.shape))
        batch_size = labels.shape[0]
        x = np.zeros([batch_size, w, h, 12])
        for k in range(batch_size):
            for i in range(w):
                for j in range(h):
                    x[k, i, j, labels[k, i, j]] = 1
        return x
class CamvidDataProvid(DataProvider):
    """Abstract class for cifar readers"""

    def __init__(self, save_path=None, validation_set=None,
                 validation_split=None, shuffle=None, normalization=None,
                 one_hot=True, **kwargs):
        """
        Args:
            save_path: `str`
            validation_set: `bool`.
            validation_split: `float` or None
                float: chunk of `train set` will be marked as `validation set`.
                None: if 'validation set' == True, `validation set` will be
                    copy of `test set`
            shuffle: `str` or None
                None: no any shuffling
                once_prior_train: shuffle train data only once prior train
                every_epoch: shuffle train data prior every epoch
            normalization: `str` or None
                None: no any normalization
                divide_255: divide all pixels by 255
                divide_256: divide all pixels by 256
                by_chanels: substract mean of every chanel and divide each
                    chanel data by it's standart deviation
            one_hot: `bool`, return lasels one hot encoded
        """
        save_path = "../data/CamVid/"
        self._save_path = save_path
        self.one_hot = one_hot
        #download_data_url(self.data_url, self.save_path)
        train_fnames, test_fnames, validation_fnames = self.get_filenames(self.save_path)
        print(str(train_fnames)+str(test_fnames)+str(validation_fnames))
        

        # add train and validations datasets
        images = self.read_camvid(train_fnames)
        #plt.imshow(images['label'][100])
        #plt.show()
        
        
        if validation_set is not None and validation_split is not None:
            split_idx = int(images.shape[0] * (1 - validation_split))
            self.train = CamvidDataSet(
                images=images[:split_idx], labels=labels[:split_idx],
                n_classes=self.n_classes, shuffle=shuffle,
                normalization=normalization,
                augmentation=self.data_augmentation)
            self.validation = CamvidDataSet(
                images=images[split_idx:], labels=labels[split_idx:],
                n_classes=self.n_classes, shuffle=shuffle,
                normalization=normalization,
                augmentation=False)
        else:
        	  print('Reading training images')
            self.train = CamvidDataSet(
                images=np.array(images['image']), labels=np.array(images['label']),
                n_classes=self.n_classes, shuffle=shuffle,
                normalization=normalization,
                augmentation=self.data_augmentation)

        print(str(self.train.num_examples))
        
        # add test set
        images = self.read_camvid(test_fnames)
        #images = images[0:500]
        #labels = labels[0:500]
        self.test = CamvidDataSet(
            images=np.array(images['image']), labels=np.array(images['label']),
            shuffle=None, n_classes=self.n_classes,
            normalization=normalization,
            augmentation=False)
        print(str(self.test.num_examples))

        images = self.read_camvid(validation_fnames)

        self.validation =  CamvidDataSet(
            images=np.array(images['image']), labels=np.array(images['label']),
            shuffle=None, n_classes=self.n_classes,
            normalization=normalization,
            augmentation=False)

        if validation_set and not validation_split:
            self.validation = self.test
        

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = os.path.join(
                tempfile.gettempdir(), 'cifar%d' % self.n_classes)
        return self._save_path

    @property
    def data_url(self):
        """Return url for downloaded data depends on cifar class"""
        data_url = ('url to download the data')
        return data_url

    @property
    def data_shape(self):
	#360,480
        return (224, 224, 3)

    @property
    def n_classes(self):
        return self._n_classes

    def get_filenames(self, save_path):
        """Return two lists of train and test filenames for dataset"""
        raise NotImplementedError
    
    def normalized(self, rgb):
        #return rgb/255.0
        norm=np.zeros((rgb.shape[0], rgb.shape[1], 3),np.float32)

        b=rgb[:,:,0]
        g=rgb[:,:,1]
        r=rgb[:,:,2]

        norm[:,:,0]=cv2.equalizeHist(b)
        norm[:,:,1]=cv2.equalizeHist(g)
        norm[:,:,2]=cv2.equalizeHist(r)

        return norm

    def one_hot_it(self, labels,w,h):
        x = np.zeros([w,h,12])
        for i in range(w):
            for j in range(h):
                x[i,j,labels[i][j]]=1
        return x
    def random_crop(self,img, mask, 
        crop_size, patch_step=(20, 20),
        teacher_pred=None, teacher_soft=None):
        
        # Get image size
        image_size = img.shape

        # Find horizontal cropping indices
        if image_size[0] > crop_size[0]:
            bound1 = np.arange(0, image_size[0] - crop_size[0],
                               patch_step[0]).astype("int32")
            bound2 = np.arange(crop_size[0], image_size[0],
                               patch_step[0]).astype("int32")
            g1 = list(zip(bound1, bound2))
            random_state.shuffle(g1)
            lr = g1[0]
        else:
            lr = (0, image_size[0])

        # Find vertical cropping indices
        if image_size[1] > crop_size[1]:
            bound3 = np.arange(0, image_size[1] - crop_size[1],
                               patch_step[1]).astype("int32")
            bound4 = np.arange(crop_size[1], image_size[1],
                               patch_step[1]).astype("int32")
            g2 = list(zip(bound3, bound4))
            random_state.shuffle(g2)
            ud = g2[0]
        else:
            ud = (0, image_size[1])

        # Crop image and mask
        img = img[lr[0]:lr[1], ud[0]:ud[1]]
        mask = mask[lr[0]:lr[1], ud[0]:ud[1]]

        rval = [img, mask]

        if teacher_pred is not None:
            pred = teacher_pred[lr[0]:lr[1], ud[0]:ud[1]]
            rval = rval + [pred]
        if teacher_soft is not None:
            soft = teacher_soft[lr[0]:lr[1], ud[0]:ud[1]]
            rval = rval + [soft]

        return rval
    def randomCrop(self, img, mask, width, height):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == mask.shape[0]
        assert img.shape[1] == mask.shape[1]
        x = random_state.randint(0, img.shape[1] - width)
        y = random_state.randint(0, img.shape[0] - height)
        
        cropImage =np.zeros((width, width, 3),np.float32)
        for i in range(0,3):
        	temp = img[:,:,i]
        	cropImage(:,:,i) = temp[y:y+height, x:x+width]
        #img = img[y:y+height, x:x+width]
        mask = mask[y:y+height, x:x+width]
        return cropImage, mask

    def read_camvid(self, filenames):
        camviddata={}
        camviddata['image'] = []
        camviddata['label'] = []
        i=0
        with open(filenames, 'r') as file:
            csv_reader = csv.reader(file, delimiter= ' ')
            for row in csv_reader:
                #print(row[0])
                #img = self.normalized(cv2.resize(cv2.imread(row[0], cv2.IMREAD_COLOR),(int(224),int(224))))
                #img = cv2.resize(cv2.imread(row[0], cv2.IMREAD_COLOR),(int(224),int(224)))
                img , mask = self.randomCrop(self.normalized(cv2.imread(row[0], cv2.IMREAD_COLOR)),cv2.imread(row[1],cv2.IMREAD_UNCHANGED),int(224),int(224))
                
                
                
                
                #camviddata['label'].append(cv2.resize(cv2.imread(row[1], cv2.IMREAD_COLOR),(int(224),int(224)))[:,:,0])
                #mask = cv2.resize(cv2.imread(row[1],cv2.IMREAD_UNCHANGED),(int(224),int(224)))
                
                #d = self.random_crop(img,mask,(224,224))
                
                camviddata['image'].append(img)
                camviddata['label'].append(mask)
                i = i+1
                    
                    
        '''
        if self.n_classes == 10:
            labels_key = b'labels'
        elif self.n_classes == 100:
            labels_key = b'fine_labels'

        images_res = []
        labels_res = []
        for fname in filenames:
            with open(fname, 'rb') as f:
                images_and_labels = pickle.load(f)
            images = images_and_labels[b'data']
            images = images.reshape(-1, 3, 32, 32)
            images = images.swapaxes(1, 3).swapaxes(1, 2)
            images_res.append(images)
            labels_res.append(images_and_labels[labels_key])
        images_res = np.vstack(images_res)
        labels_res = np.hstack(labels_res)
        if self.one_hot:
            labels_res = self.labels_to_one_hot(labels_res)
        '''
        return camviddata
 

class CamvidDataProvider(CamvidDataProvid):
    _n_classes = 12
    data_augmentation = True

    def get_filenames(self, save_path):
        train_filenames = os.path.join(save_path, 'train_edited.txt')
        test_filenames = os.path.join(save_path, 'test_edited.txt')
        validation_filenames = os.path.join(save_path, 'validation_edited.txt')
        return train_filenames, test_filenames, validation_filenames

if __name__ == '__main__':
    # some sanity checks for Cifar data providers
    c10_provider = CamvidDataProvider(
        validation_set=True)
