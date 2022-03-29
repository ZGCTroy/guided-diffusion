# from collections import defaultdict
import numpy as np
import sklearn.svm
import tensorflow as tf
import dnnlib.tflib as tflib
import dnnlib
from tensorflow.python.client import device_lib

from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from sklearn.metrics import roc_curve
# ---------------- Changed Here --------------------
from sklearn.metrics import auc as cal_auc
import argparse
import os
# import torch
from scripts.fid_score import calculate_fid_given_paths

classifier_urls = [
    'https://drive.google.com/uc?id=1Q5-AI6TwWhCVM7Muu4tBM7rp5nG_gmCX', # celebahq-classifier-00-male.pkl
    'https://drive.google.com/uc?id=1Q5c6HE__ReW2W8qYAXpao68V1ryuisGo', # celebahq-classifier-01-smiling.pkl
    'https://drive.google.com/uc?id=1Q7738mgWTljPOJQrZtSMLxzShEhrvVsU', # celebahq-classifier-02-attractive.pkl
    'https://drive.google.com/uc?id=1QBv2Mxe7ZLvOv1YBTLq-T4DS3HjmXV0o', # celebahq-classifier-03-wavy-hair.pkl
    'https://drive.google.com/uc?id=1QIvKTrkYpUrdA45nf7pspwAqXDwWOLhV', # celebahq-classifier-04-young.pkl
    'https://drive.google.com/uc?id=1QJPH5rW7MbIjFUdZT7vRYfyUjNYDl4_L', # celebahq-classifier-05-5-o-clock-shadow.pkl
    'https://drive.google.com/uc?id=1QPZXSYf6cptQnApWS_T83sqFMun3rULY', # celebahq-classifier-06-arched-eyebrows.pkl
    'https://drive.google.com/uc?id=1QPgoAZRqINXk_PFoQ6NwMmiJfxc5d2Pg', # celebahq-classifier-07-bags-under-eyes.pkl
    'https://drive.google.com/uc?id=1QQPQgxgI6wrMWNyxFyTLSgMVZmRr1oO7', # celebahq-classifier-08-bald.pkl
    'https://drive.google.com/uc?id=1QcSphAmV62UrCIqhMGgcIlZfoe8hfWaF', # celebahq-classifier-09-bangs.pkl
    'https://drive.google.com/uc?id=1QdWTVwljClTFrrrcZnPuPOR4mEuz7jGh', # celebahq-classifier-10-big-lips.pkl
    'https://drive.google.com/uc?id=1QgvEWEtr2mS4yj1b_Y3WKe6cLWL3LYmK', # celebahq-classifier-11-big-nose.pkl
    'https://drive.google.com/uc?id=1QidfMk9FOKgmUUIziTCeo8t-kTGwcT18', # celebahq-classifier-12-black-hair.pkl
    'https://drive.google.com/uc?id=1QthrJt-wY31GPtV8SbnZQZ0_UEdhasHO', # celebahq-classifier-13-blond-hair.pkl
    'https://drive.google.com/uc?id=1QvCAkXxdYT4sIwCzYDnCL9Nb5TDYUxGW', # celebahq-classifier-14-blurry.pkl
    'https://drive.google.com/uc?id=1QvLWuwSuWI9Ln8cpxSGHIciUsnmaw8L0', # celebahq-classifier-15-brown-hair.pkl
    'https://drive.google.com/uc?id=1QxW6THPI2fqDoiFEMaV6pWWHhKI_OoA7', # celebahq-classifier-16-bushy-eyebrows.pkl
    'https://drive.google.com/uc?id=1R71xKw8oTW2IHyqmRDChhTBkW9wq4N9v', # celebahq-classifier-17-chubby.pkl
    'https://drive.google.com/uc?id=1RDn_fiLfEGbTc7JjazRXuAxJpr-4Pl67', # celebahq-classifier-18-double-chin.pkl
    'https://drive.google.com/uc?id=1RGBuwXbaz5052bM4VFvaSJaqNvVM4_cI', # celebahq-classifier-19-eyeglasses.pkl
    'https://drive.google.com/uc?id=1RIxOiWxDpUwhB-9HzDkbkLegkd7euRU9', # celebahq-classifier-20-goatee.pkl
    'https://drive.google.com/uc?id=1RPaNiEnJODdr-fwXhUFdoSQLFFZC7rC-', # celebahq-classifier-21-gray-hair.pkl
    'https://drive.google.com/uc?id=1RQH8lPSwOI2K_9XQCZ2Ktz7xm46o80ep', # celebahq-classifier-22-heavy-makeup.pkl
    'https://drive.google.com/uc?id=1RXZM61xCzlwUZKq-X7QhxOg0D2telPow', # celebahq-classifier-23-high-cheekbones.pkl
    'https://drive.google.com/uc?id=1RgASVHW8EWMyOCiRb5fsUijFu-HfxONM', # celebahq-classifier-24-mouth-slightly-open.pkl
    'https://drive.google.com/uc?id=1RkC8JLqLosWMaRne3DARRgolhbtg_wnr', # celebahq-classifier-25-mustache.pkl
    'https://drive.google.com/uc?id=1RqtbtFT2EuwpGTqsTYJDyXdnDsFCPtLO', # celebahq-classifier-26-narrow-eyes.pkl
    'https://drive.google.com/uc?id=1Rs7hU-re8bBMeRHR-fKgMbjPh-RIbrsh', # celebahq-classifier-27-no-beard.pkl
    'https://drive.google.com/uc?id=1RynDJQWdGOAGffmkPVCrLJqy_fciPF9E', # celebahq-classifier-28-oval-face.pkl
    'https://drive.google.com/uc?id=1S0TZ_Hdv5cb06NDaCD8NqVfKy7MuXZsN', # celebahq-classifier-29-pale-skin.pkl
    'https://drive.google.com/uc?id=1S3JPhZH2B4gVZZYCWkxoRP11q09PjCkA', # celebahq-classifier-30-pointy-nose.pkl
    'https://drive.google.com/uc?id=1S3pQuUz-Jiywq_euhsfezWfGkfzLZ87W', # celebahq-classifier-31-receding-hairline.pkl
    'https://drive.google.com/uc?id=1S6nyIl_SEI3M4l748xEdTV2vymB_-lrY', # celebahq-classifier-32-rosy-cheeks.pkl
    'https://drive.google.com/uc?id=1S9P5WCi3GYIBPVYiPTWygrYIUSIKGxbU', # celebahq-classifier-33-sideburns.pkl
    'https://drive.google.com/uc?id=1SANviG-pp08n7AFpE9wrARzozPIlbfCH', # celebahq-classifier-34-straight-hair.pkl
    'https://drive.google.com/uc?id=1SArgyMl6_z7P7coAuArqUC2zbmckecEY', # celebahq-classifier-35-wearing-earrings.pkl
    'https://drive.google.com/uc?id=1SC5JjS5J-J4zXFO9Vk2ZU2DT82TZUza_', # celebahq-classifier-36-wearing-hat.pkl
    'https://drive.google.com/uc?id=1SDAQWz03HGiu0MSOKyn7gvrp3wdIGoj-', # celebahq-classifier-37-wearing-lipstick.pkl
    'https://drive.google.com/uc?id=1SEtrVK-TQUC0XeGkBE9y7L8VXfbchyKX', # celebahq-classifier-38-wearing-necklace.pkl
    'https://drive.google.com/uc?id=1SF_mJIdyGINXoV-I6IAxHB_k5dxiF6M-', # celebahq-classifier-39-wearing-necktie.pkl
]

# 16 glass 21 gender 25 bread 32 smiling 36 hat 10 blond hair 40 young 9 black hair
# 19 makeup 37 lipstick 5 bald 18 gray hair
att_dict={
    '16':'glass',
    '21':'gender',
    '25':'bread',
    '32':'smiling',
    '40':'young',
}
pkl_dict={
    '16':'celebahq-classifier-19-eyeglasses.pkl',
    '21':'celebahq-classifier-00-male.pkl',
    '25':'celebahq-classifier-27-no-beard.pkl',
    '32':'celebahq-classifier-01-smiling.pkl',
    '40':'celebahq-classifier-04-young.pkl',
}

import pickle
# import config
def open_file_or_url(file_or_url):
    return open(file_or_url, 'rb')

def load_pkl(file_or_url):
    with open_file_or_url(file_or_url) as file:
        return pickle.load(file, encoding='latin1')

sess = tflib.init_tf()
def get_acc(opt, img, label, cond, classifier_name):

    import torch
    classifier_path = os.path.join(opt.base_path, pkl_dict[classifier_name]) 
    # img = img.to(f'cuda:{opt.gpus}')
    with tf.device(f'/gpu:{opt.gpus}'):
        images = tf.placeholder(tf.float32, shape=(opt.batch_size,3,256,256))
        classifier = load_pkl(classifier_path)
        classifier.tranable = False
        logits = classifier.get_output_for(images, None)
        predictions = tf.stop_gradient(tf.nn.softmax(tf.concat([logits, -logits], axis=1)))
        # tf.stop_gradient(predictions)

    batches = torch.split(img, opt.batch_size, dim=0)
    if len(batches) * opt.batch_size != img.shape[0]:
        batches = batches[:-1]

    preds = []
    with tf.device(f'/gpu:{opt.gpus}'):
        for idx, img_batch in enumerate(batches):
            output = sess.run(predictions, feed_dict={images:img_batch})
            # print(output)
            preds.append(output[:, 1])

    preds = np.concatenate(preds, axis=0)
    label, preds = np.array(label[:preds.shape[0], cond]), np.array(preds)
    acc = accuracy_score(label, preds > 0.5)

    # sess.close()
    return acc

import glob
def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dir_base', type=str, default="", help='base path for predict dir')
    parser.add_argument('--save_name', type=str, default="", help='name for predict log')
    
    parser.add_argument('--base_path', type=str, default="", help='base pretrain path for dir')
    parser.add_argument('--fid_stat_path', type=str, default="", help="preprocessed fid stat path")

    parser.add_argument('--batch_size', type=int, default=8, help='predict batch size')
    parser.add_argument('--gpus', type=str, default="0", help='KD_weigtht')
    
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    import torch

    opt.log_dir = os.path.join(opt.dir_base, opt.save_name)
    
    glob_path = os.path.join(opt.log_dir, 'samples*.npz')
    log_paths = glob.glob(glob_path)

    img_list = []
    acc_dict = {}
    tot_acc = 0

    for path in log_paths:
        data = np.load(path)
        img, label = data['arr_0'], data['arr_1']
        cond = os.path.basename(path).split('_')[1]
        classifier = os.path.basename(path).split('_')[2]

        # if cond == '0':
        #     continue
        # print(cond, classifier)
        img = torch.from_numpy(img.transpose(0,3,1,2)).float().div(255)
        img = (img - 0.5) * 2. 

        att_key = classifier + '_' + att_dict[classifier] + '_acc'
        acc_dict[att_key] = get_acc(opt, img, label, int(cond), classifier)
        tot_acc += acc_dict[att_key]

        img_list.append(img)
    
    acc_dict['tot_acc'] = tot_acc / len(log_paths)

    imgs = torch.cat(img_list, dim=0)
    
    acc_dict['fid'] = calculate_fid_given_paths(opt.fid_stat_path, imgs,
                                            50, True, 2048)

    acc_dict = {k:float(v) for k,v in acc_dict.items()}
    
    import yaml
    # d = {'a':1, 0:2, 'sd':{0:1,2:{3:1}}}
    save_path = os.path.join(opt.log_dir, 'results_dict.yaml')
    fp = open(save_path, 'w')
    fp.write(yaml.dump(acc_dict))
    fp.close()

if __name__ == "__main__":
    main()
