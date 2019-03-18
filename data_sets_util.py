import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import os
import os.path
import glob
import numpy as np
import random
from torch.utils.data import DataLoader
from torchvision import transforms
#---------------------------------------------------------------------------------------
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
#---------------------------------------------------------------------------------------
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
#---------------------------------------------------------------------------------------
def default_loader(path):
    return Image.open(path).convert('RGB')
#---------------------------------------------------------------------------------------
def shuffle_list(list_data,indices):
    list_array = np.asarray(list_data, dtype=object)
    list_array = list_array[indices]
    return list_array.tolist()
#---------------------------------------------------------------------------------------
def make_Dsprites_Im2Im(dir_path, scale_labelF=0, orient_labelF=1):
    #---------------------
    # 0: Color: white
    # 1: Shape: square, ellipse, heart
    # 2: Scale: 6 values linearly spaced in [0.5, 1]
    # 3: Orientation: 40 values in [0, 2 pi]
    # 4: Position X: 32 values in [0, 1]
    # 5: Position Y: 32 values in [0, 1]

    # --------------
    images = []
    labels = []
    orients_labels = []
    dataset_zip = np.load(dir_path+'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')

    images = dataset_zip['imgs']
    latents_values = dataset_zip['latents_values']
    latents_classes = dataset_zip['latents_classes']
    shape_labels = np.array(latents_classes[:,4])
    if scale_labelF == 0 :
        scale_labels = np.array(latents_values[:,2])
    else:
        scale_labels = anp.array(latents_classes[:,2])

    if orient_labelF == 0 :
        orients_labels = np.array(latents_values[:,3])
    else:
        orients_labels = np.array(latents_classes[:,3])

    ort_classes =  np.array(latents_classes[:,3])
    print(len(latents_classes[:,3]))
    print(ort_classes.shape)
    print(images.shape)
    left_range = np.append(np.linspace(0, 10, 10, endpoint=False),np.linspace(20, 30, 10, endpoint=False)).astype('int') 
    left_indx = np.isin(ort_classes,left_range)
    right_range = np.append(np.linspace(10, 20, 10, endpoint=False),np.linspace(30, 40, 10, endpoint=False)).astype('int')
    right_indx = np.isin(ort_classes,right_range)
    
    imgs_A = images[left_indx,:,:]
    shape_A = shape_labels[left_indx]
    scale_A = scale_labels[left_indx]
    orients_A = orients_labels[left_indx]
   
    ind_shuffle_A = np.arange(0,imgs_A.shape[0])
    np.random.shuffle(ind_shuffle_A)
    train_ind = int(imgs_A.shape[0] * 0.7)

    imgs_train_A =imgs_A[ind_shuffle_A[:train_ind],:,:] 
    shape_train_A = shape_A[ind_shuffle_A[:train_ind]]
    scale_train_A = scale_A[ind_shuffle_A[:train_ind]]
    orients_train_A = orients_A[ind_shuffle_A[:train_ind]]
    train_A = {'img':imgs_train_A , 'shape':shape_train_A , 'scale':scale_train_A , 'orit':orients_train_A }

    imgs_test_A =imgs_A[ind_shuffle_A[train_ind:],:,:]
    shape_test_A = shape_A[ind_shuffle_A[train_ind:]]
    scale_test_A = scale_A[ind_shuffle_A[train_ind:]]
    orients_test_A = orients_A[ind_shuffle_A[train_ind:]]
    test_A = {'img':imgs_test_A , 'shape':shape_test_A , 'scale':scale_test_A , 'orit':orients_test_A }

    print(imgs_A.shape)

    imgs_B = images[right_indx,:,:]
    shape_B = shape_labels[right_indx]
    scale_B = scale_labels[right_indx]
    orients_B = orients_labels[right_indx]

    ind_shuffle_B = np.arange(0,imgs_B.shape[0])
    np.random.shuffle(ind_shuffle_B)
    train_ind = int(imgs_B.shape[0] * 0.7)

    imgs_train_B =imgs_B[ind_shuffle_B[:train_ind],:,:]
    shape_train_B = shape_B[ind_shuffle_B[:train_ind]]
    scale_train_B = scale_B[ind_shuffle_B[:train_ind]]
    orients_train_B = orients_B[ind_shuffle_B[:train_ind]]
    train_B = {'img':imgs_train_B , 'shape':shape_train_B , 'scale':scale_train_B , 'orit':orients_train_B }

    imgs_test_B =imgs_B[ind_shuffle_B[train_ind:],:,:]
    shape_test_B = shape_B[ind_shuffle_B[train_ind:]]
    scale_test_B = scale_B[ind_shuffle_B[train_ind:]]
    orients_test_B = orients_B[ind_shuffle_B[train_ind:]]
    test_B = {'img':imgs_test_B , 'shape':shape_test_B , 'scale':scale_test_B , 'orit':orients_test_B }
    return train_A, train_B, test_A, test_B

#---------------------------------------------------------------------------------------
class ImageDspritesDataSetIm2Im(data.Dataset):
    def __init__(self,data_A, data_B,transform=None,loader=default_loader):
        self.data_A = data_A
        self.data_B = data_B
        self.ind_shuffle_A = np.arange(0,len(data_A['img']))
        np.random.shuffle(self.ind_shuffle_A)
        self.ind_shuffle_B = np.arange(0,len(data_B['img']))
        np.random.shuffle(self.ind_shuffle_B)
        self.transform = transform
        self.len_A = len(data_A['img'])
        self.len_B = len(data_B['img'])
        self.loader = loader
    def __getitem__(self,index):
        # A data 
        img_a = self.data_A['img'][self.ind_shuffle_A[index%self.len_A]][:,:]*255
        img_a = Image.fromarray(img_a.astype('uint8'))
        shape_a = self.data_A['shape'][self.ind_shuffle_A[index%self.len_A]]
        scale_a = self.data_A['scale'][self.ind_shuffle_A[index%self.len_A]]
        ort_a = self.data_A['orit'][self.ind_shuffle_A[index%self.len_A]]
        if self.transform is not None:
            img_a = self.transform(img_a)
        img_a = torch.from_numpy(np.array(img_a))
        # B data  
        img_b = self.data_B['img'][self.ind_shuffle_B[index%self.len_B]][:,:]*255
        img_b = Image.fromarray(img_b.astype('uint8'))
        shape_b = self.data_B['shape'][self.ind_shuffle_B[index%self.len_B]]
        scale_b = self.data_B['scale'][self.ind_shuffle_B[index%self.len_B]]
        ort_b = self.data_B['orit'][self.ind_shuffle_B[index%self.len_B]]
        if self.transform is not None:
            img_b = self.transform(img_b)
        img_b = torch.from_numpy(np.array(img_b))
        return img_a, shape_a, scale_a, ort_a, img_b, shape_b, scale_b, ort_b

    def __len__(self):
        return max(self.len_A,self.len_B )

#---------------------------------------------------------------------------------------
def create_Dsprites_im2imloaders(num_workers,batch_size, height, width,new_size,crop=False):
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]

    #transform_list = [transforms.ToTensor(),] 
    transform_list = [transforms.RandomCrop((height, width))] + transform_list if crop else transform_list
    transform_list = [transforms.Scale(new_size)] + transform_list if new_size is not None else transform_list
    #transform_list = [transforms.RandomHorizontalFlip()] + transform_list if train else transform_list
    transform = transforms.Compose(transform_list)
    train_dataset, test_dataset =  create_Dsprites_im2imDatasets(transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    return train_loader, test_loader
#---------------------------------------------------------------------------------------
def create_Dsprites_im2imDatasets(trans):
    dir_path = '/share/jproject/fg508/eman/domainAdapt/GANStyle/VAE_tut/dsprites-dataset/'
    train_A, train_B , test_A , test_B = make_Dsprites_Im2Im(dir_path)
    train_dataset = ImageDspritesDataSetIm2Im(train_A, train_B,transform=trans)
    test_dataset = ImageDspritesDataSetIm2Im(test_A, test_B,transform=trans)
    return train_dataset, test_dataset
#---------------------------------------------------------------------------------------
#train_loader, test_loader= create_Dsprites_im2imloaders(1,2,128,128,128)
#for i,(img_a, shape_a, scale_a, ort_a, img_b, shape_b, scale_b, ort_b) in enumerate(train_loader):
#    print(' img ')
#    break 

#dir_path = '/share/jproject/fg508/eman/domainAdapt/GANStyle/VAE_tut/dsprites-dataset/'
#make_Dsprites_Im2Im(dir_path)

#---------------------------------------------------------------------------------------
def make_folder_imgs_3DChairsIm2ImOrt(dir):
    images_A = []
    images_B = []
    ort_labl_A = []
    ort_labl_B = []

    tdata = ['000','011','023','034','046', \
             '058','069','081','092','104', \
             '116','127','139','150','162', \
             '174','185','197','209','220', \
             '232','243','255','267','278', \
             '290','301','313','325','336', \
             '348']
    A_data = len(tdata)/2
    indx = 0
    for i in range(len(tdata)):
        id1 = '%03d'%indx
        id2 = tdata[i]
        if i <= A_data :
            images_A.append(dir+'/image_'+id1+'_p020_t'+id2+'_r096.png')
            ort_labl_A.append(i)
        else :
            images_B.append(dir+'/image_'+id1+'_p020_t'+id2+'_r096.png')
            ort_labl_B.append(i)
        indx += 1

    indx += 1
    for i in range(len(tdata)):
        id1 = '%03d'%indx
        id2 = tdata[i]
        if i >= A_data :
            images_A.append(dir+'/image_'+id1+'_p030_t'+id2+'_r096.png')
            ort_labl_A.append(i)
        else:
            images_B.append(dir+'/image_'+id1+'_p030_t'+id2+'_r096.png')
            ort_labl_B.append(i)
        indx += 1

    return images_A, images_B , ort_labl_A, ort_labl_B
#---------------------------------------------------------------------------------------
def sample_dataset_chairs_list(dir_path):
    train_im_a = []
    train_im_b = []
    test_im_a = []
    test_im_b = []

    train_ort_a = []
    train_ort_b = []
    test_ort_a = []
    test_ort_b = []

    train_cl_a = []
    train_cl_b = []
    test_cl_a = []
    test_cl_b = []
    #
    ind = 0
    classes = []
    for sub_root, sub_dirs, sub_files in os.walk(dir_path):
        if ind == 0:
            classes = sub_dirs
            break
    classes =  sorted(classes, key=str.lower)
    p_train = 0.7
    for cl_i,cl in enumerate(classes):
        if random.random() <= p_train :
            # train sample
            images_a, images_b,ort_labl_A, ort_labl_B = make_folder_imgs_3DChairsIm2ImOrt(dir_path+'/'+cl+'/renders/')
            train_im_a.extend(images_a)
            train_im_b.extend(images_b)
            train_ort_a.extend(ort_labl_A)
            train_ort_b.extend(ort_labl_B)
            train_cl_a.extend([cl_i]*len(images_a))
            train_cl_b.extend([cl_i]*len(images_a))
        else :
            # test sample
            images_a, images_b, ort_labl_A, ort_labl_B = make_folder_imgs_3DChairsIm2ImOrt(dir_path+'/'+cl+'/renders/')
            test_im_a.extend(images_a)
            test_im_b.extend(images_b )
            test_ort_a.extend(ort_labl_A)
            test_ort_b.extend(ort_labl_B)
            test_cl_a.extend([cl_i]*len(images_a))
            test_cl_b.extend([cl_i]*len(images_a))
   
    train_A = {'img':train_im_a, 'ort': train_ort_a , 'cl':train_cl_a }
    train_B = {'img':train_im_b, 'ort': train_ort_b , 'cl':train_cl_b }
    test_A = {'img':test_im_a, 'ort': test_ort_a , 'cl':test_cl_a }
    test_B = {'img':test_im_b, 'ort': test_ort_b , 'cl':test_cl_b }
    return train_A, train_B , test_A , test_B 
#---------------------------------------------------------------------------------------
class ImageChairsDataSetIm2Im(data.Dataset):
    def __init__(self,data_A, data_B,transform=None,loader=default_loader):
        self.data_A = data_A 
        self.data_B = data_B 
        self.ind_shuffle_A = np.arange(0,len(data_A['img']))
        np.random.shuffle(self.ind_shuffle_A)
        self.ind_shuffle_B = np.arange(0,len(data_B['img']))
        np.random.shuffle(self.ind_shuffle_B)
        self.transform = transform
        self.len_A = len(data_A['img'])
        self.len_B = len(data_B['img']) 
        self.loader = loader
    def __getitem__(self,index):
        # A data 
        path = self.data_A['img'][self.ind_shuffle_A[index%self.len_A]]
        img_a = self.loader(path)
        l_a = self.data_A['ort'][self.ind_shuffle_A[index%self.len_A]]
        ort_a = self.data_A['cl'][self.ind_shuffle_A[index%self.len_A]]
        if self.transform is not None:
            img_a = self.transform(img_a)
        img_a = torch.from_numpy(np.array(img_a))
        # B data  
        path = self.data_B['img'][self.ind_shuffle_B[index%self.len_B]]
        img_b = self.loader(path)
        l_b = self.data_B['ort'][self.ind_shuffle_B[index%self.len_B]]
        ort_b = self.data_B['cl'][self.ind_shuffle_B[index%self.len_B]]
        if self.transform is not None:
            img_b = self.transform(img_b)
        img_b = torch.from_numpy(np.array(img_b))
        return img_a, l_a, ort_a, img_b, l_b, ort_b

    def __len__(self):
        return max(self.len_A,self.len_B )
#---------------------------------------------------------------------------------------
def create_chairs_im2imloaders(num_workers,batch_size, height, width,new_size,crop=False):    
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]

    #transform_list = [transforms.ToTensor(),] 
    transform_list = [transforms.RandomCrop((height, width))] + transform_list if crop else transform_list
    transform_list = [transforms.Scale(new_size)] + transform_list if new_size is not None else transform_list
    #transform_list = [transforms.RandomHorizontalFlip()] + transform_list if train else transform_list
    transform = transforms.Compose(transform_list)
    train_dataset, test_dataset =  create_chairs_im2imDatasets(transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    return train_loader, test_loader
#---------------------------------------------------------------------------------------
def create_chairs_im2imDatasets(trans):
    dir_path = '/share/jproject/fg508/eman/domainAdapt/GANStyle/VAE_tut/rendered_chairs/'
    train_A, train_B , test_A , test_B = sample_dataset_chairs_list(dir_path)   
    train_dataset = ImageChairsDataSetIm2Im(train_A, train_B,transform=trans)
    test_dataset = ImageChairsDataSetIm2Im(test_A, test_B,transform=trans)
    return train_dataset, test_dataset  
#---------------------------------------------------------------------------------------
#train_loader, test_loader= create_chairs_im2imloaders(1,2,128,128,128)
#for i,(img_a, l_a, ort_a, img_b, l_b, ort_b) in enumerate(train_loader):
#    print(i)
#    break 

'''
def make_folder_imgs_3DChairsIm2Im(dir):
    images_A = []
    images_B = []
    
    tdata = ['000','011','023','034','046', \
             '058','069','081','092','104', \
             '116','127','139','150','162', \
             '174','185','197','209','220', \
             '232','243','255','267','278', \
             '290','301','313','325','336', \
             '348']
    A_data = len(tdata)/2 
    indx = 0
    for i in range(len(tdata)):
        id1 = '%03d'%indx
        id2 = tdata[i]
        if i <= A_data : 
            images_A.append(dir+'/image_'+id1+'_p020_t'+id2+'_r096.png')
        else :
            images_B.append(dir+'/image_'+id1+'_p020_t'+id2+'_r096.png')
        indx += 1

    indx += 1
    for i in range(len(tdata)):
        id1 = '%03d'%indx
        id2 = tdata[i]
        if i >= A_data :
            images_A.append(dir+'/image_'+id1+'_p030_t'+id2+'_r096.png')
        else:
            images_B.append(dir+'/image_'+id1+'_p030_t'+id2+'_r096.png')
        indx += 1

    return images_A, images_B

#---------------------------------------------------------------------------------------
def make_folder_imgs_list3DChairs(dir):
    images = [] 
    tdata = ['000','011','023','034','046', \
             '058','069','081','092','104', \
             '116','127','139','150','162', \
             '174','185','197','209','220', \
             '232','243','255','267','278', \
             '290','301','313','325','336', \
             '348']
    ort_labl = [] 
    # image_058_p030_t301_r096.png
    indx = 0  
    for i in range(len(tdata)):
        id1 = '%03d'%indx
        id2 = tdata[i]
        images.append(dir+'/image_'+id1+'_p020_t'+id2+'_r096.png')  
        ort_labl.append(i)
        indx += 1
    indx += 1
    for i in range(len(tdata)):
        id1 = '%03d'%indx
        id2 = tdata[i]
        images.append(dir+'/image_'+id1+'_p030_t'+id2+'_r096.png')
        ort_labl.append(i)
        indx += 1

    return images,ort_labl
#---------------------------------------------------------------------------------------
def make_folder_imgs_list(dir):
    images = []
    for ext in IMG_EXTENSIONS :
        images.extend(glob.glob(dir+'/*'+ext))
    random.shuffle(images,random.random)
    return images
#---------------------------------------------------------------------------------------
def sample_3dshape_im2im(dir_path,out_dir):
    train_im_a = []
    train_im_b = []
    test_im_a = []
    test_im_b = []
    ind = 0
    classes = []
    for sub_root, sub_dirs, sub_files in os.walk(dir_path):
        if ind == 0:
            classes = sub_dirs
            break
    classes =  sorted(classes, key=str.lower)
    p_train = 0.7
    for cl_i,cl in enumerate(classes):
        if random.random() <= p_train :
            # train sample
            images_a, images_b = make_folder_imgs_3DChairsIm2Im(dir_path+'/'+cl+'/renders/')
            train_im_a.extend(images_a)
            train_im_b.extend(images_b)
        else :
            # test sample
            images_a, images_b = make_folder_imgs_3DChairsIm2Im(dir_path+'/'+cl+'/renders/')
            test_im_a.extend(images_a)
            test_im_b.extend(images_b )

#---------------------------------------------------------------------------------------
def sample_dataset_im2im(dir_path,out_dir):
    train_im_a = []
    train_im_b = []
    test_im_a = []
    test_im_b = []
    #
    ind = 0
    classes = []
    for sub_root, sub_dirs, sub_files in os.walk(dir_path):
        if ind == 0:
            classes = sub_dirs
            break
    classes =  sorted(classes, key=str.lower)
    p_train = 0.7 
    for cl_i,cl in enumerate(classes):
        if random.random() <= p_train :
            # train sample
            images_a, images_b = make_folder_imgs_3DChairsIm2Im(dir_path+'/'+cl+'/renders/')
            train_im_a.extend(images_a)
            train_im_b.extend(images_b) 
        else :
            # test sample
            images_a, images_b = make_folder_imgs_3DChairsIm2Im(dir_path+'/'+cl+'/renders/')
            test_im_a.extend(images_a)
            test_im_b.extend(images_b )
    print(' train_im_a ')
    print(len(train_im_a))
    print(' train_im_b ')
    print(len(train_im_b))
    print(' test_im_a ')
    print(len(test_im_a))
    print(' test_im_b ')
    print(len(test_im_b))

    # copy data 
    data_path_1 = out_dir + '/trainA/'
    for ind, img_dir in enumerate(train_im_a):
        print(' ind ')
        print(ind)
        l_token = img_dir.split('/')
        img_name = l_token[-1]
        folder_name = l_token[len(l_token) - 4 ]
        res_path = data_path_1 + folder_name + '/' 
        if not os.path.exists(res_path): 
            os.mkdir(res_path)
        img_load = default_loader(img_dir)      
        img_load.save(res_path+'/'+img_name)

    data_path_1 = out_dir + '/trainB/'
    for ind, img_dir in enumerate(train_im_b):
        print(' ind ')
        print(ind)
        l_token = img_dir.split('/')
        img_name = l_token[-1]
        folder_name = l_token[len(l_token) - 4 ]
        res_path = data_path_1 + folder_name + '/'
        if not os.path.exists(res_path):
            os.mkdir(res_path)
        img_load = default_loader(img_dir)
        img_load.save(res_path+'/'+img_name)

    data_path_1 = out_dir + '/testA/'
    for ind, img_dir in enumerate(test_im_a):
        print(' ind ')
        print(ind)
        l_token = img_dir.split('/')
        img_name = l_token[-1]
        folder_name = l_token[len(l_token) - 4 ]
        res_path = data_path_1 + folder_name + '/'
        if not os.path.exists(res_path):
            os.mkdir(res_path)
        img_load = default_loader(img_dir)
        img_load.save(res_path+'/'+img_name)

    data_path_1 = out_dir + '/testB/'
    for ind, img_dir in enumerate(test_im_b):
        print(' ind ')
        print(ind)
        l_token = img_dir.split('/')
        img_name = l_token[-1]
        folder_name = l_token[len(l_token) - 4 ]
        res_path = data_path_1 + folder_name + '/'
        if not os.path.exists(res_path):
            os.mkdir(res_path)
        img_load = default_loader(img_dir)
        img_load.save(res_path+'/'+img_name)
 
    #l_token = train_im_a[0].split('/')
    #img_name = l_token[-1]
    #folder_name = l_token[len(l_token) - 2 ]
    #data_path_1 = out_dir + '/trainA/' 
    #print(' tokenizer ')
    #print(l_token)    
    #data_path_1 = out_dir + '/trainA/'
    #data_path_1 = out_dir + '/testA/'

    
#---------------------------------------------------------------------------------------
def make_dataset3DChairs(dir_path):
    images = []
    labels = []
    orients_labels = []
    ind = 0
    classes = []
    for sub_root, sub_dirs, sub_files in os.walk(dir_path):
        if ind == 0:
            classes = sub_dirs
            break
    classes =  sorted(classes, key=str.lower)
    for cl_i,cl in enumerate(classes):
        l_imgs,ort_labl = make_folder_imgs_list3DChairs(dir_path+'/'+cl+'/renders/')
        l_labels = [cl_i]*len(l_imgs)
        images.extend(l_imgs)
        labels.extend(l_labels)
        orients_labels.extend(ort_labl)
    
    ind_shuffle = np.arange(0,len(images))
    ind_shuffle = np.random.shuffle(ind_shuffle)

    shuffle_list(images,ind_shuffle)
    shuffle_list(labels,ind_shuffle)
    shuffle_list(orients_labels,ind_shuffle)    

    return images, labels, orients_labels 
#---------------------------------------------------------------------------------------
class ImageChairsDataSet(data.Dataset):
    def __init__(self,dir_path ,transform=None, loader=default_loader):
        images, labels, orients_labels =  make_dataset3DChairs(dir_path)
        self.images = images 
        self.labels = labels 
        self.orients_labels = orients_labels
        self.transform = transform
        self.loader = loader
    def __getitem__(self, index):
        path = self.images[index]
        img = self.loader(path)
        l = self.labels[index]
        ort_lb = self.orients_labels[index]
        if self.transform is not None:
            img = self.transform(img)
        img = torch.from_numpy(np.array(img))
        return img,l,ort_lb 

    def __len__(self):
        return len(self.labels)
        
#---------------------------------------------------------------------------------------
def make_Dsprites_Im2Im(dir_path, out_path,scale_labelF=0, orient_labelF=1):
    #---------------------
    # 0: Color: white
    # 1: Shape: square, ellipse, heart
    # 2: Scale: 6 values linearly spaced in [0.5, 1]
    # 3: Orientation: 40 values in [0, 2 pi]
    # 4: Position X: 32 values in [0, 1]
    # 5: Position Y: 32 values in [0, 1]
    
    # --------------
    images = []
    labels = []
    orients_labels = []
    dataset_zip = np.load(dir_path+'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')

    images = dataset_zip['imgs']
    latents_values = dataset_zip['latents_values']
    latents_classes = dataset_zip['latents_classes']
    shape_labels = latents_classes[:,4]
    if scale_labelF == 0 :
        scale_labels = latents_values[:,2]
    else:
        scale_labels = latents_classes[:,2]

    if orient_labelF == 0 :
        orients_labels = latents_values[:,3]
    else:
        orients_labels = latents_classes[:,3]
    
    p_train = 0.7
    pathes_str = ['/trainA/','/trainB/','/testA/','/testB/']
    for i in range(0,len(orients_labels)): 
        print(i )       
        if random.random() <= p_train :
            # train imgs 
            ort_l = latents_classes[i,3]
            print(' ort_l ')
            print(ort_l)
            if (ort_l >= 0 and ort_l <= 9) or (ort_l >= 20 and ort_l <= 29) :
                # trainA 
                im_res = Image.fromarray(images[i])
                im_res.save(out_path+pathes_str[0]+'Image_'+str(i)+'.png')
            else :   
                # trainB
                im_res = Image.fromarray(images[i])
                im_res.save(out_path+pathes_str[1]+'Image_'+str(i)+'.png')
        else :
            # test imgs
            ort_l = latents_classes[i,2]
            if (ort_l >= 0 and ort_l <= 9) or (ort_l >= 20 and ort_l <= 29) :
                # testA 
                im_res = Image.fromarray(images[i])
                im_res.save(out_path+pathes_str[2]+'Image_'+str(i)+'.png')
            else :
                # testB 
                im_res = Image.fromarray(images[i])
                im_res.save(out_path+pathes_str[3]+'Image_'+str(i)+'.png')
 
    print(' res latent ')
    print(latents_classes[737200:737280,3])
    print(len(latents_classes[:,2]))
    
    print(np.unique(latents_classes[:,0]))
    print(np.unique(latents_classes[:,1]))
    print(np.unique(latents_classes[:,2]))
    print(np.unique(latents_classes[:,3]))
    print(np.unique(latents_classes[:,4]))
    print(np.unique(latents_classes[:,5]))
    
#---------------------------------------------------------------------------------------
def make_datasetDsprites(dir_path, scale_labelF=0, orient_labelF=1):
    
    # 0: Color: white
    # 1: Shape: square, ellipse, heart
    # 2: Scale: 6 values linearly spaced in [0.5, 1]
    # 3: Orientation: 40 values in [0, 2 pi]
    # 4: Position X: 32 values in [0, 1]
    # 5: Position Y: 32 values in [0, 1]

    images = []
    labels = []
    orients_labels = []
    dataset_zip = np.load(dir_path+'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')

    images = dataset_zip['imgs']
    latents_values = dataset_zip['latents_values']
    latents_classes = dataset_zip['latents_classes']
    shape_labels = latents_classes[:,4]
    if scale_labelF == 0 :
        scale_labels = latents_values[:,3]
    else:
        scale_labels = latents_classes[:,3]
    
    if orient_labelF == 0 :
        orients_labels = latents_values[:,2]
    else:
        orients_labels = latents_classes[:,2]

    return images, shape_labels, scale_labels, orients_labels
     
#---------------------------------------------------------------------------------------
class DspritesDataSet(data.Dataset):
    def __init__(self,dir_path ,transform=None,scale_labelF=0, orient_labelF=1):
        images, shape_labels, scale_labels, orients_labels =  make_datasetDsprites(dir_path, scale_labelF=scale_labelF, orient_labelF=orient_labelF)
        self.images = images
        self.shape_labels = shape_labels
        self.scale_labels = scale_labels
        self.orients_labels = orients_labels
        self.ind_shuffle = np.arange(0,len(images))
        np.random.shuffle(self.ind_shuffle)
        self.transform = transform 
    def __getitem__(self, index):
        index = self.ind_shuffle[index]
        img = self.images[index]
        shape_l = self.shape_labels[index]
        scale_l = self.scale_labels[index]
        ort_lb = self.orients_labels[index]
        if self.transform is not None:
            img = self.transform(img)
        img = torch.from_numpy(np.array(img))
        return img, shape_l, scale_l, ort_lb

    def __len__(self):
        return len(self.shape_labels)

#---------------------------------------------------------------------------------------
dir_path = '/share/jproject/fg508/eman/domainAdapt/GANStyle/VAE_tut/dsprites-dataset/'  
out_dir = '/share/jproject/fg508/eman/domainAdapt/GANStyle/dsprites_im2Im/'
make_Dsprites_Im2Im(dir_path,out_dir)

#dir_path = '/share/jproject/fg508/eman/domainAdapt/GANStyle/VAE_tut/rendered_chairs/'
#out_dir = '/share/jproject/fg508/eman/domainAdapt/GANStyle/CarsChairs3D_Im2Im/'
#sample_dataset_im2im(dir_path,out_dir)
'''
'''
dir_path = '/share/jproject/fg508/eman/domainAdapt/GANStyle/VAE_tut/dsprites-dataset/'
dataset_test = DspritesDataSet(dir_path)
test_loader = torch.utils.data.DataLoader(dataset_test,batch_size=64,shuffle=False,num_workers = 1)

for i, (img, shape_l, scale_l, ort_lb) in enumerate(test_loader):
    print(' img ')
    print(img.shape)
    print(' shape l ')
    print(shape_l.shape)
    print(' scale_l ')
    print(scale_l.shape)
    print(' ort_lb ')
    print(ort_lb.shape) 
    break 
   
dir_path = '/share/jproject/fg508/eman/domainAdapt/GANStyle/VAE_tut/rendered_chairs/'
dataset_test = ImageChairsDataSet(dir_path)
test_loader = torch.utils.data.DataLoader(dataset_test,batch_size=32,shuffle=False,num_workers = 1) 
for j, (img,l,ort_lb) in enumerate(test_loader):
    print(' img ')
    print(img.shape)
    print(' l ' )
    print(l.shape)
    print('ort_lb')
    print(ort_lb.shape)
    break
'''
 
'''
dir_path = '/share/jproject/fg508/eman/domainAdapt/GANStyle/VAE_tut/dsprites-dataset/' 
images, shape_labels, scale_labels, orients_labels = make_datasetDsprites(dir_path)
print('images')
print(images.shape)
print(' shape_labels ')
print(shape_labels.shape)
print(' scale_labels ')
print(scale_labels.shape)
print(' orients_labels ')
print(orients_labels.shape)
'''

#dir_path = '/share/jproject/fg508/eman/domainAdapt/GANStyle/VAE_tut/rendered_chairs/'
#make_dataset3DChairs(dir_path)
#---------------------------------------------------------------------------------------



