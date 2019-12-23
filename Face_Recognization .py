import numpy as np
from keras.utils import np_utils
from keras.layers import Dense,Dropout,Convolution2D,Flatten,MaxPooling2D
from keras.models import Sequential,load_model
import cv2 as cv
import os
import face_recognition
import _pickle as pickle
from collections import defaultdict

'''
numpy：处理数据
keras：搭建神经网络框架
cv2：处理摄像头和图片大小
os：训练时打开文件&导入图片
face_recognition：检测图片中的人脸
json：保存和家在字典
defaultdict：创建字典
'''






'''
错误名单：
1711453:libpng warning: iCCP: known incorrect sRGB profile！
1711465：部分图片像素格式错误
1711468：后5张图片像素格式错误
1711479：图片像素格式错误
1711492：命名格式错误，图片错误
1711515：部分图片像素格式错误
1711521：部分图片像素格式错误，例如10216
1711522：部分图片像素格式错误
1711525：命名格式错误，图片像素格式错误
'''


'''
选做内容：视频识别、摄像头识别、算法优化
报告+源码+测试效果
'''

class Face_Recognization():
    def __init__(self,dataset_path):
        #init datas & name directionary
        self.image_all=[]
        self.label_all=[]
        self.image_test=[]
        self.label_test=[]

        self.k2number_dir=defaultdict(int)
        self.number2name_dir=defaultdict(int)
        self.dataset_path=dataset_path
        self.k=0

    #讲学号和名字对应
    def create_name_dir(self):
        number_list=[
            1613329,            1710402,            1710675,            1710696,
            1711448,            1711449,            1711450,            1711452,
            1711453,            1711454,            1711455,            1711456,
            1711457,            1711458,            1711459,            1711460,
            1711461,            1711462,            1711463,            1711464,
            1711465,            1711466,            1711467,            1711468,
            1711469,            1711470,            1711471,            1711472,
            1711474,            1711475,            1711476,            1711477,
            1711478,            1711479,            1711480,            1711481,
            1711482,            1711483,            1711484,            1711485,
            1711486,            1711487,            1711488,            1711489,
            1711491,            1711492,            1711494,            1711495,
            1711496,            1711497,            1711498,            1711499,
            1711500,            1711501,            1711502,            1711503,
            1711504,            1711505,            1711506,            1711507,
            1711508,            1711510,            1711511,            1711512,
            1711513,            1711514,            1711515,            1711516,
            1711517,            1711518,            1711519,            1711520,
            1711521,            1711522,            1711523,            1711524,
            1711525,            1711526,            1711527,            1711528,
            1711529,            1711530,            1711533,            1711535,
            1711536,            1713402,            1910771,            1911613,
            1911614            ]
        name_list=[
            '尹潇婧',        '郎睿博',        '杨宇航',        '郭惠珂',
            '巴赫',        '包云鹏',        '曹瑞',        '陈斌',
            '陈文庆',        '崔同悦',        '董硕',        '范嘉骏',
            '郭俑鑫',        '纪文轩',        '冀彦东',        '贾熠辰',
            '简宇豪',        '姜顺昌',        '靳葳',        '李铭赫',
            '李睿',        '李天奕',        '梁骞',        '刘冲',
            '刘光耀',        '刘静阳',        '刘权',        '刘翔',
            '刘育铭',        '刘卓',        '鲁传恒',        '鲁航',
            '鲁扬',        '孟洋',        '聂韧',        '潘比炜',
            '裴逸飞',        '蒲肃',        '普显东',        '骞晨旭',
            '钱前',        '申晨宇',        '苏宁洋',        '孙思毓',
            '唐光盛',        '陶京星',        '王杨',        '王禹',
            '王岳霖',        '吴岳洋',        '武霆威',        '肖遥',
            '徐圆凯',        '杨澳',        '杨定也',        '杨帆',
            '战祥昕',        '张力元',        '张千一',        '张硕',
            '张詇',        '张兆鹏',        '赵宇',        '郑嘉文',
            '周亦瑾',        '朱迪',        '曹谦谦',        '董伊梦',
            '冯玉昕',        '龚佳怡',        '郭欣然',        '郭珍慧',
            '姜文颖',        '李屹梅',        '梁佳宸',        '凌潇',
            '刘可佳',        '罗茜',        '苗艺馨',        '石京京',
            '覃洁媛',        '韦知辛',        '翟晓琳',        '张恩平',
            '张艺凡',        '郭元白',        '李瑾媛',        '杭一洲',
            '何嘉笙'        ]
        for i in range(len(number_list)):
            self.number2name_dir[number_list[i]]=name_list[i]
        #保存字典
        with open('number2name_dir', 'wb') as fp:
            pickle.dump(self.number2name_dir, fp)


    #数据预处理
    def load_data(self):
        black = np.zeros([128,128,3], dtype='uint8')
        white = np.ones([128,128,3], dtype='uint8') * 255
        #dataset_path : data
        for first_dir_item in os.listdir(self.dataset_path):
            print('loading'+first_dir_item+'......')
            self.k+=1
            #first_dir_item: 1711xxx
            second_dir_item = os.path.abspath(os.path.join(self.dataset_path, first_dir_item))
            temp=0
            for dir_item in os.listdir(second_dir_item):
                temp+=1
                if dir_item.endswith('.png'):
                    full_path = os.path.abspath(os.path.join(second_dir_item, dir_item))
                    image= cv.imread(full_path)
                    if image.shape != (128, 128, 3):
                        image = cv.resize(image, (128, 128), interpolation=cv.INTER_CUBIC)
                    w1 = cv.addWeighted(image, 1.1, white, -0.1, 10)
                    w2 = cv.addWeighted(image, 1.3, white, -0.3, 10)
                    b1 = cv.addWeighted(image, 1.1, black, -0.1, 10)
                    b2 = cv.addWeighted(image, 1.3, black, -0.3, 10)
                    #700的训练集，300的测试机
                    if temp<700:
                        self.image_all.append(image)
                        self.label_all.append(self.k)
                        self.image_all.append(w1)
                        self.label_all.append(self.k)
                        self.image_all.append(w2)
                        self.label_all.append(self.k)
                        self.image_all.append(b1)
                        self.label_all.append(self.k)
                        self.image_all.append(b2)
                        self.label_all.append(self.k)
                    else:
                        self.image_test.append(image)
                        self.label_test.append(self.k)
            self.k2number_dir[self.k]=first_dir_item

        self.label_all = np.array(self.label_all)
        self.image_all = np.array(self.image_all)

        self.label_test = np.array(self.label_test)
        self.image_test = np.array(self.image_test)
        #one hot
        self.label_all=np_utils.to_categorical(self.label_all,num_classes=self.k+1)
        self.label_test=np_utils.to_categorical(self.label_test,num_classes=self.k+1)
        #保存字典
        with open('k2number_dir', 'wb') as fp:
            pickle.dump(self.k2number_dir, fp)

    #定义神经网络
    def Net_model(self):
        #定义顺序模型
        self.FR_model=Sequential()
        #第一个卷积层
        #input_shape 输入平面
        #filters 卷积核/滤波器
        #kernel_size 卷积窗口大小
        #strides 步长
        #padding 方式为same/valid
        #activation 激活函数
        self.FR_model.add(Convolution2D(
            input_shape=(128,128,3),
            filters = 32,
            kernel_size = 3,
            strides=1,
            padding = 'same',
            activation = 'relu',
        ))
        #第一个池化层
        self.FR_model.add(MaxPooling2D(
            pool_size = 2,
            strides = 2,
            padding = 'same'
        ))
        #第一个dropout
        self.FR_model.add(Dropout(0.5))

        #第二个卷积层
        self.FR_model.add(Convolution2D(
            filters = 64,
            kernel_size = 3,
            strides = 1,
            padding = 'same',
            activation = 'relu'
        ))
        #第二个池化层
        self.FR_model.add(MaxPooling2D(
            pool_size=2,
            strides=2,
            padding='same'))
        #第二个dropout
        self.FR_model.add(Dropout(0.5))
        #把第二个池化层的输出扁平化为1层
        self.FR_model.add(Flatten())
        #全连接层
        self.FR_model.add(Dense(512,activation='relu'))
        self.FR_model.add(Dense(128, activation='relu'))
        self.FR_model.add(Dense(64,activation='relu'))
        self.FR_model.add(Dense(32,activation='relu'))
        #输出层
        self.FR_model.add(Dense(85,activation='softmax'))
        #定义优化器
        #定义优化器，loss function，训练过程中计算
        self.FR_model.compile(optimizer='adagrad',loss='categorical_crossentropy',metrics=['accuracy'])


    #训练模型
    def train_model(self):
        self.FR_model.fit(self.image_all,self.label_all,batch_size=32,epochs=40)



    #评估模型
    def test(self):
        loss,accuracy=self.FR_model.evaluate(self.image_test,self.label_test)
        print('test loss',loss)
        print('test accuracy',accuracy)


    #保存模型
    def save_model(self):
        self.FR_model.save('model_v2.h5')
        self.FR_model.save_weights('model_weights_v2.h5')


    #加载模型
    def load_model(self):
        #self.FR_model=load_model('model_v2.h5')
        self.FR_model.load_weights('model_weights_v2.h5')


    #加载字典
    def load_dir(self):
        with open('k2number_dir', 'rb') as fp:
            self.k2number_dir = pickle.load(fp)
        with open('number2name_dir', 'rb') as fp:
            self.number2name_dir = pickle.load(fp)

    def capture(self):
        '''
        打开摄像头，读取帧，检测该帧图像中的人脸，并进行剪切、缩放
        生成图片满足以下格式：
        1.灰度图，后缀为 .png
        2.图像大小相同
        params:
            data:指定生成的人脸数据的保存路径
        '''

        # 打开摄像头
        camera = cv.VideoCapture(0)
        cv.namedWindow('Capture Face')
        # 计数

        while (True):
            rec_lo=[]
            rec_rd=[]
            num_all = []
            name_all = []
            # 读取一帧图像
            ret, frame = camera.read()
            if ret:
                #face_locations获取照片中人脸的位置
                face_locations = face_recognition.face_locations(frame)
                for face_num in range(len(face_locations)):
                    f_l=face_locations[face_num]
                    x1 = min(f_l[1], f_l[3])
                    x2 = max(f_l[1], f_l[3])
                    y1 = min(f_l[0], f_l[2])
                    y2 = max(f_l[0], f_l[2])
                    image = frame[y1:y2, x1:x2]
                    image = cv.resize(image, (128, 128), interpolation=cv.INTER_CUBIC)
                    num,name=self.predict(image.reshape(1,128,128,3))
                    rec_lo.append((x1,y1))
                    rec_rd.append((x2,y2))
                    num_all.append(num)
                    name_all.append(name)
                #绘图
                if len(face_locations)!=0:
                    for i in range(len(face_locations)):
                        cv.rectangle(frame, rec_lo[i], rec_rd[i], (0, 255, 0), 3)
                        font = cv.FONT_HERSHEY_SIMPLEX
                        if len(name_all) > 0:
                            print(num_all[i])
                            print(name_all[i])
                            # cv.putText(frame, text='%s:%s'%(num_all[i],name_all[i]), org=(rec_lo[i][0] + 5, rec_lo[i][1] + 5), fontFace=font, fontScale=1, thickness=1,
                            #         lineType=cv.LINE_AA, color=(0, 255, 0))
                            cv.putText(frame, text='%s'%(num_all[i]), org=(rec_lo[i][0] + 5, rec_lo[i][1] + 5), fontFace=font, fontScale=1, thickness=1,
                                    lineType=cv.LINE_AA, color=(0, 255, 0))

                cv.imshow('Capture Face', frame)
                # 按下q键退出
                if cv.waitKey(100) & 0xff == ord('q'):
                    break
        camera.release()
        cv.destroyAllWindows()

    #预测图片中人的名字-，返回学号和名字
    def predict(self,pic):
        p=self.FR_model.predict(pic)
        k = np.argmax(p, axis=1)
        print(k[0])
        num = self.k2number_dir[k[0]]
        name = self.number2name_dir[int(num)]
        return num,name

    #检测test集中的图片(必须是128,128,3的图片)
    def predit_test(self,pic_name):
        pic = cv.imread(pic_name)
        pic=np.array(pic)
        pic=pic.reshape(1,128,128,3)
        p=self.FR_model.predict(pic)
        k = np.argmax(p, axis=1)
        num = self.k2number_dir[int(k)]
        name = self.number2name_dir[int(num)]
        print('学号：%s；名字：%s'%(num,name))

    #输入原始图片，输出被predict后的图片（任意大小的图片）
    def predit_people(self,pic):
        rec_lo = []
        rec_rd = []
        num_all = []
        name_all = []
        face_locations = face_recognition.face_locations(pic)
        for face_num in range(len(face_locations)):
            f_l = face_locations[face_num]
            x1 = min(f_l[1], f_l[3])
            x2 = max(f_l[1], f_l[3])
            y1 = min(f_l[0], f_l[2])
            y2 = max(f_l[0], f_l[2])
            image = pic[y1:y2, x1:x2]
            image = cv.resize(image, (128, 128), interpolation=cv.INTER_CUBIC)
            num, name = self.predict(image.reshape(1, 128, 128, 3))
            rec_lo.append((x1, y1))
            rec_rd.append((x2, y2))
            num_all.append(num)
            name_all.append(name)
        #绘图
        if len(face_locations) != 0:
            for i in range(len(face_locations)):
                cv.rectangle(pic, rec_lo[i], rec_rd[i], (0, 255, 0), 3)
                font = cv.FONT_HERSHEY_SIMPLEX
                if len(name_all) > 0:
                    # cv.putText(pic, text='%s:%s' % (num_all[i], name_all[i]),
                    #            org=(rec_lo[i][0] + 5, rec_lo[i][1] + 5), fontFace=font, fontScale=1, thickness=1,
                    #            lineType=cv.LINE_AA, color=(0, 255, 0))
                    cv.putText(pic, text='%s' % (num_all[i]),
                               org=(rec_lo[i][0] + 5, rec_lo[i][1] + 5), fontFace=font, fontScale=1, thickness=1,
                               lineType=cv.LINE_AA, color=(0, 255, 0))
        return pic


    #读取视频
    def read_video(self,video_path):
        cap = cv.VideoCapture(video_path)
        self.pic_list=[]
        while (cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
            cv.imshow('reading_video',frame)
            self.size=frame.shape
            frame_new=self.predit_people(frame)
            self.pic_list.append(frame_new)
        print('read video done!')
        cap.release()
        cv.destroyAllWindows()


    #合成视频
    def synthesis_video(self):
        fps = 24  # 视频每秒24帧
        size = (self.size[0],self.size[1] )  # 需要转为视频的图片的尺寸
        # 可以使用cv2.resize()进行修改
        video = cv.VideoWriter("predicted_video.avi", cv.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
        # 视频保存在当前目录下
        for item in self.pic_list:
            video.write(item)
        video.release()
        cv.destroyAllWindows()

    #讲图片合并成视频，供识别使用
    def pic2video(self):
        filelist = []
        for i in range(10701, 10999):
            p = cv.imread('%d.png' % i)
            print(i)
            filelist.append(p)
        print(filelist)

        fps = 24  # 视频每秒24帧
        size = (128, 128)  # 需要转为视频的图片的尺寸
        # 可以使用cv2.resize()进行修改
        fourcc1 = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
        # fourcc2 = cv.VideoWriter_fourcc('I', '4', '2', '0')
        video = cv.VideoWriter("VideoTest1.mp4", fourcc1, fps, size)
        # 视频保存在当前目录下
        for item in filelist:
            video.write(item)
        video.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    #对模型进行train和test，运行下面这部分
    '''
    FR=Face_Recognization(dataset_path='data')
    FR.create_name_dir()
    FR.load_data()
    FR.Net_model()
    FR.train_model()
    FR.save_model()
    FR.test()
    '''

    #加载训练好的模型并用准备好的图片进行测试，运行下面这部分
    '''
    FR=Face_Recognization(dataset_path='data')
    FR.load_dir()
    FR.Net_model()
    FR.load_model()
    FR.predit_test('1.png')
    FR.predit_test('2.png')
    FR.predit_test('3.png')
    FR.predit_test('4.png')
    FR.predit_test('5.png')
    FR.predit_test('6.png')
    FR.predit_test('7.png')
    FR.predit_test('8.png')
    FR.predit_test('9.png')
    p=cv.imread('temp.jpg')
    pp=FR.predit_people(p)
    cv.imshow('temp',pp)
    cv.waitKey(0)
    '''


    #加载训练好的模型并读取摄像头，运行下面这部分
    FR=Face_Recognization(dataset_path='data')
    FR.load_dir()
    FR.Net_model()
    FR.load_model()
    FR.capture()

    # 加载训练好的模型并处理视频流，运行下面这部分
    '''
    FR=Face_Recognization(dataset_path='data')
    FR.load_dir()
    FR.Net_model()
    FR.load_model()
    FR.read_video('VideoTest1.mp4')
    FR.synthesis_video()
    '''







