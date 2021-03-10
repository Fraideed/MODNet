from PySide2.QtGui import QImage, QPixmap
from PySide2.QtWidgets import QApplication, QMessageBox, QGraphicsPixmapItem, QGraphicsScene
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import QFile
from urllib import request
import cv2
import os
import sys,os
import PySide2

# dirname = os.path.dirname(PySide2.__file__)
# plugin_path = os.path.join(dirname, 'plugins', 'platforms')
# os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
# print(plugin_path)

#import tensorflow as tf

import numpy as np
import csv
import tqdm
from tkinter import *
import tkinter.filedialog
from shutil import copyfile
from cffi.setuptools_ext import execfile

data_folder_name = '/home/jazz/temp'
'''
data_path_name = 'cv'
pic_path_name = 'pic'
cv_path_name = 'fer2013'
csv_file_name = 'fer2013.csv'
ckpt_name = 'cnn_emotion_classifier.ckpt'
model_path_name = 'cnn_inuse'
casc_name = 'haarcascade_frontalface_alt.xml'
cv_path = os.path.join(data_folder_name, data_path_name, cv_path_name)
csv_path = os.path.join(cv_path, csv_file_name)
ckpt_path = os.path.join(data_folder_name, data_path_name, model_path_name, ckpt_name)
casc_path = os.path.join(data_folder_name, data_path_name, casc_name)
pic_path = os.path.join(data_folder_name, data_path_name, pic_path_name)

channel = 1
default_height = 48
default_width = 48
confusion_matrix = False
use_advanced_method = True
emotion_labels = ['angry', 'disgust:', 'fear', 'happy', 'sad', 'surprise', 'neutral']
num_class = len(emotion_labels)


# config=tf.ConfigProto(log_device_placement=True)
sess = tf.Session()

saver = tf.train.import_meta_graph(ckpt_path+'.meta')
saver.restore(sess, ckpt_path)
graph = tf.get_default_graph()
name = [n.name for n in graph.as_graph_def().node]
print(name)
x_input = graph.get_tensor_by_name('x_input:0')
dropout = graph.get_tensor_by_name('dropout:0')
logits = graph.get_tensor_by_name('project/output/logits:0')


def advance_image(images_):
    rsz_img = []
    rsz_imgs = []
    for image_ in images_:
        rsz_img.append(image_)
    rsz_imgs.append(np.array(rsz_img))
    rsz_img = []
    for image_ in images_:
        rsz_img.append(np.reshape(cv2.resize(image_[2:45, :], (default_height, default_width)),
                                  [default_height, default_width, channel]))
    rsz_imgs.append(np.array(rsz_img))
    rsz_img = []
    for image_ in images_:
        rsz_img.append(np.reshape(cv2.resize(cv2.flip(image_, 1), (default_height, default_width)),
                                  [default_height, default_width, channel]))
    rsz_imgs.append(np.array(rsz_img))
    return rsz_imgs


def produce_result(images_):
    images_ = np.multiply(np.array(images_), 1. / 255)
    if use_advanced_method:
        rsz_imgs = advance_image(images_)
    else:
        rsz_imgs = [images_]
    pred_logits_ = []
    for rsz_img in rsz_imgs:
        pred_logits_.append(sess.run(tf.nn.softmax(logits), {x_input: rsz_img, dropout: 1.0}))
    return np.sum(pred_logits_, axis=0)


def produce_results(images_):
    results = []
    pred_logits_ = produce_result(images_)
    pred_logits_list_ = np.array(np.reshape(np.argmax(pred_logits_, axis=1), [-1])).tolist()
    for num in range(num_class):
        results.append(pred_logits_list_.count(num))
    result_decimals = np.around(np.array(results) / len(images_), decimals=3)
    return results, result_decimals


def produce_confusion_matrix(images_list_, total_num_):
    total = []
    total_decimals = []
    for ii, images_ in enumerate(images_list_):
        results, result_decimals = produce_results(images_)
        total.append(results)
        total_decimals.append(result_decimals)
        print(results, ii, ":", result_decimals[ii])
        print(result_decimals)
    sum = 0
    for i_ in range(num_class):
        sum += total[i_][i_]
    print('acc: {:.3f} %'.format(sum * 100. / total_num_))
    print('Using ', ckpt_name)


def predict_emotion(image_):
    image_ = cv2.resize(image_, (default_height, default_width))
    image_ = np.reshape(image_, [-1, default_height, default_width, channel])
    return produce_result(image_)[0]


def face_detect(casc_path_=casc_path):
    if os.path.isfile(casc_path_):
        face_casccade_ = cv2.CascadeClassifier(casc_path_)
        img_ = cv2.imread(pic_path)
        img_gray_ = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
        # face detection
        faces = face_casccade_.detectMultiScale(
            img_gray_,
            scaleFactor=1.1,
            minNeighbors=1,
            minSize=(30, 30),
        )
        return faces, img_gray_, img_
    else:
        print("There is no {} in {}".format(casc_name, casc_path_))
'''

class Stats:

    def __init__(self):
        # 从文件中加载UI定义
        qfile_stats = QFile("classifier.ui")
        qfile_stats.open(QFile.ReadOnly)
        qfile_stats.close()
        self.ui = QUiLoader().load(qfile_stats)

        self.ui.button1.clicked.connect(self.click_button1)
        self.ui.button2.clicked.connect(self.click_button2)

    def click_button1(self):
        global pic_path
        pic_path = tkinter.filedialog.askopenfilename()
        root = tkinter.Tk()  # 创建一个Tkinter.Tk()实例
        root.withdraw()  # 将Tkinter.Tk()实例隐藏
        print(pic_path)

        link = pic_path
        # request.urlretrieve(link,'F:/Pycharm/PyCharm Community Edition 2020.2.3/ServiceInnovate/CNN/inputs')
        copyfile(pic_path,'inputs/01.jpg')

        img = cv2.imread(pic_path)  # 读取图像
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换图像通道
        x = img.shape[1]  # 获取图像大小
        y = img.shape[0]
        self.zoomscale = 1 # 图片放缩尺度
        frame = QImage(img, x, y, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
        self.item.setScale(self.zoomscale)
        self.scene = QGraphicsScene()  # 创建场景
        self.scene.addItem(self.item)
        self.ui.graphicsView.setScene(self.scene)  # 将场景添加至视图

    def click_button2(self):

        os.system('python inference.py')
        #execfile('F:\\Pycharm\\PyCharm Community Edition 2020.2.3\\ServiceInnovate\\CNN\\inference.py')
        #path = 'F:\Pycharm\PyCharm Community Edition 2020.2.3\ServiceInnovate\CNN\inference.py'
        #with open(path, 'r', encoding='utf-8')
'''
                  
        if not confusion_matrix:
            print(pic_path)
            faces, img_gray, img = face_detect()
            spb = img.shape
            sp = img_gray.shape
            height = sp[0]
            width = sp[1]
            size = 600
            emotion_pre_dict = {}
            face_exists = 0
            for (x, y, w, h) in faces:
                face_exists = 1
                face_img_gray = img_gray[y:y + h, x:x + w]
                results_sum = predict_emotion(face_img_gray)  # face_img_gray
                for i, emotion_pre in enumerate(results_sum):
                    emotion_pre_dict[emotion_labels[i]] = emotion_pre
                # 输出所有情绪的概率
                print(emotion_pre_dict)
                label = np.argmax(results_sum)
                emo = emotion_labels[int(label)]
                print('Emotion : ', emo)
                # 输出最大概率的情绪
                # 使框的大小适应各种像素的照片
                t_size = 2
                ww = int(spb[0] * t_size / 300)
                www = int((w + 10) * t_size / 100)
                www_s = int((w + 20) * t_size / 100) * 2 / 5
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), ww)
                cv2.putText(img, emo, (x + 2, y + h - 2), cv2.FONT_HERSHEY_SIMPLEX,www_s, (255, 0, 255), thickness=www, lineType=1)
                # img_gray full face     face_img_gray part of face
            if face_exists:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换图像通道
                x = img.shape[1]  # 获取图像大小
                y = img.shape[0]
                self.zoomscale = 1  # 图片放缩尺度
                frame = QImage(img, x, y, QImage.Format_RGB888)
                pix = QPixmap.fromImage(frame)
                self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
                # self.item.setScale(self.zoomscale)
                self.scene = QGraphicsScene()  # 创建场景
                self.scene.addItem(self.item)
                self.ui.graphicsView.setScene(self.scene)  # 将场景添加至视图
        if confusion_matrix:
            with open(csv_path, 'r') as f:
                csvr = csv.reader(f)
                header = next(csvr)
                rows = [row for row in csvr]
                val = [row[:-1] for row in rows if row[-1] == 'PublicTest']
                tst = [row[:-1] for row in rows if row[-1] == 'PrivateTest']
            confusion_images_total = []
            confusion_images = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
            test_set = tst
            total_num = len(test_set)
            for label_image_ in test_set:
                label_ = int(label_image_[0])
                image_ = np.reshape(np.asarray([int(p) for p in label_image_[-1].split()]),
                                    [default_height, default_width, 1])
                confusion_images[label_].append(image_)
            produce_confusion_matrix(confusion_images.values(), total_num)
'''

app = QApplication([])
stats = Stats()
stats.ui.show()
app.exec_()


