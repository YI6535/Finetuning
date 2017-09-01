
# coding: utf-8

import cv2
import numpy as np
import os
import pickle
from tqdm import tqdm
import numpy.random
import random
import argparse
import chainer
import json
import datetime
from chainer import optimizers
import chainer.functions as F
import chainer.links as L
from chainer import serializers
from chainer.functions import caffe
from chainer import cuda
from chainer import Variable
from chainer import computational_graph
import sys


# パスを扱いやすくするために home_dir(フォルダの根幹) を定義<br>



# argparse

def arg_process(argument):
    parser = argparse.ArgumentParser()
    parser.add_argument('--finetune',type=bool, default=True,
                    help = 'finetune or not')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--batchsize', '-b', default=20, type=int,
                    help = 'batchsize')
    parser.add_argument('--test_val_ratio', '-t', default = 0.15, type = float,
                    help = 'val/train_ratio')
    parser.add_argument('--n_epoch', '-n', default = 100, type = int,
                    help = 'number of epoch')
    parser.add_argument('--augment', '-a', default =True, type = bool,
                    help = 'augment or not')
    parser.add_argument('--finetune_model_lr', '-f', default =1e-4, type = float,
                    help = 'learning rate of finetune fmodel')
    parser.add_argument('--my_model_lr', '-m', default = 1e-3, type = float,
                    help = 'learning rate of model i added')
    parser.add_argument('--finetune_net', '-fn', default = 'Alexnet', type = str,
                    help = 'net i finetune')
    parser.add_argument('--output_layer', '-out', default = 'fc7', type = str,
                    help = 'output layer')
    parser.add_argument('--optimizer1', '-o1', default = 'MomentumSGD', type =str,
                    help = 'optimizer used for finetune model')
    parser.add_argument('--optimizer2', '-o2', default = 'MomentumSGD', type = str,
                    help = 'optimizer used for my model')
    parser.add_argument('--rotate','-r',default=True,type=bool,
                    help='rotate augmentation')
    parser.add_argument('--flip_x','-fx',default=True,type=bool,
                    help='flip_x augmentation')
    parser.add_argument('--flip_y','-fy',default=True,type=bool,
                    help='flip_y augmentation')
    parser.add_argument('--translation','-trans',default=True,type=bool,
                    help='translation augmentation')
    parser.add_argument('--gaussian_noise','-gau',default=False,type=bool,
                    help='gaussian noise augmentation')
    parser.add_argument('--variance','-v',default=False,type=bool,
                    help='variance augmentation')
    parser.add_argument('--zoom','-z',default=False,type=bool,
                    help='zoom augmentation')
                    
    args = parser.parse_args()
    return args


# モデルグラフ生成用のデータ入力

def write_computational_graph(loss):
    with open(home_dir + '/output/graph.dot', 'w') as o:
        o.write(chainer.computational_graph.build_computational_graph((loss, )).dump())
    with open(home_dir + '/output/graph.wo_split.dot', 'w') as o:
        g = chainer.computational_graph.build_computational_graph((loss, ), remove_split=True)
        o.write(g.dump())
    print('graph generated')


#　accuracy・loss のログを書き込む関数

def write_log(type):
    if type == 'train':
        ss = json.dumps({'type' : type, 'samples': n_imgs_trained,
                                       'accuracy': sum_accuracy/sum_n, 'loss': sum_loss/sum_n}) + '\n'
    if type == 'inv':
        ss = json.dumps({'type': type, 'samples': n_imgs_trained,
                                        'accuracy': inv_acc, 'loss': inv_loss}) + '\n'
    if type == 'val':
        ss = json.dumps({'type': type, 'samples': n_imgs_trained,
                                        'accuracy': val_acc, 'loss': val_loss}) + '\n'
    logfile.write(ss)
    logfile.flush()
    print(ss)


# 計算条件の保存

def log_calc(args):
    global logfile, today
    today = str(datetime.datetime.today())
    today = today.replace(' ', ',')
    today = today.replace(',', '_')
    today = today.replace('.', '_')
    today = today.replace(':', '_')
    today = today.replace('/', '_')
    logfile = open(home_dir + '/output/{}.log'.format(today), 'a')
    calc_cond = json.dumps({'gpu': args.gpu,
                                                 'batchsize' : args.batchsize,
                                                 'test_val_ratio': args.test_val_ratio,
                                                 'n_epoch': args.n_epoch,
                                                 'augment': args.augment,
                                                 'finetune_model_lr': args.finetune_model_lr,
                                                 'my_model_lr': args.my_model_lr,
                                                 'finetune_net': args.finetune_net,
                                                 'output_layer': args.output_layer,
                                                 'optimizer1': args.optimizer1,
                                                 'optimizer2': args.optimizer2,
                                                 'rotate':args.rotate,
                                                 'flip_x':args.flip_x,
                                                 'flip_y':args.flip_y,
                                                 'translation':args.translation,
                                                 'gaussian_noise':args.gaussian_noise,
                                                 'variance':args.variance,
                                                 'zoom':args.zoom}) + '\n'
    logfile.write(calc_cond)
    logfile.flush()
    print(calc_cond)


# データの読み込み
# 数の多い上位12クラスのClassラベル・同被写体ラベル(object_label)・切り取りの行われている画像(Top12_processed)

def load_data(args):
    f = open(os.path.join(home_dir + '/data/Top12_label.pkl'), 'rb')
    Top12_label = pickle.load(f)
    f.close()
    f = open(os.path.join(home_dir + '/data/Top12_object_label.pkl'), 'rb')
    Top12_object_label = pickle.load(f)
    f.close()
    Top12_object_label = list(Top12_object_label)
    Top12_label = list(Top12_label)

    return Top12_object_label, Top12_label


# 切り取り処理後の画像の読み込み

def Top12_processed_data(args):
    Top12_processed = np.load(os.path.join(home_dir + '/data/Top12_processed.npy'))
    return Top12_processed


# データを評価用と学習用に分ける関数
# input: test用データ比率、　output: test用,training用の(file名,Class名)と object_label のリスト

def make_train_val(args,split_rate):
    train_object_label = []
    val_object_label = []
    file_name_train = []
    file_name_test = []
    Top12_object_label, Top12_label= load_data(args)
    Top12_object_label = np.array(Top12_object_label)
    Top12_label = np.array(Top12_label)
    # file名は０から順に数字が振られている。
    file_name = np.arange(len(Top12_label))
    
    for i in range(12):
        # Top12_object_labelの要素の中でClassがiであるものを重複なくi_labelに取り出す。
        i_label = np.unique(np.array(Top12_object_label)[np.array(Top12_label) == i ])
        # i_labelの中からtest用データ分のobject_labelをi_label_testに入れる。
        i_label_test = np.random.choice(i_label, int(np.round(i_label.shape[0]*split_rate)), replace = False)
        # i_labelの中でi_label_testに入っていないものをi_label_trainに入れる。
        for j in range(len(i_label_test)):
            i_label = i_label[i_label != i_label_test[j]]
        i_label_train = i_label
        train_list = np.in1d(Top12_object_label,i_label_train)
        test_list = np.in1d(Top12_object_label,i_label_test)
        
        file_name_train += [(name, class_i) for name,class_i in zip(file_name[train_list], Top12_label[train_list])]
        file_name_test += [(name, class_i) for name,class_i in zip(file_name[test_list], Top12_label[test_list])]
        val_object_label += [object_label for object_label in Top12_object_label[test_list]]
        train_object_label += [object_label for object_label in Top12_object_label[train_list]]
    return file_name_train, file_name_test, val_object_label, train_object_label


# 評価用・学習用データの決定

def get_train_val(args):
    # random な分け方で得られた評価用・学習用データを用いる。
    train_list, val_list, val_object_label, train_object_label = make_train_val(args, args.test_val_ratio)
    #評価用データを10 倍にしてvalidationを行う。
    val_list_10 = []
    for i in val_list:
        val_list_10 += [i] * 10
    val_list = val_list_10

    return train_list, val_list, val_object_label, train_object_label


# 画像の枚数を調整する
# 学習時に各個体の画像数に大きな差があると個体に対しての過学習が大きくなり、全ての個体を同じ枚数だけ学習させようとすると、画像に対しての過学習が大きくなるため、各個体の画像数が最大枚数(max_count)＊(その個体の画像数)の平方根となるようにした。

def random_aug(train_list, train_object_label):
    transposed_train = np.array(train_list).transpose()
    aug_train_list = []
    count = 0
    values,count_num = np.unique(train_object_label,return_counts=True)
    max_count = 171
    for i in range(len(count_num)):
        n = count_num[i]
        stay_n,random_n = divmod(np.sqrt(n*max_count),n)
        random_list = np.arange(count,count+n)
        random_box = np.random.choice(random_list,int(random_n),replace=False)
        aug_train_list.append(train_list[count:count+n]*int(stay_n))
        for j in random_box:
            aug_train_list.append([train_list[j]])
        count += n
    train_list = [e for inner_list in aug_train_list for e in inner_list]
    return train_list


#epoch ごとにrandom_aug関数で作ったtrain_listを格納する。

def get_all_train(train_list ,train_object_label):
    all_aug_train = []
    for i in range(args.n_epoch):
        aug_train= random_aug(train_list, train_object_label)
        random.shuffle(aug_train)
        all_aug_train += [aug_train]
    random.shuffle(all_aug_train)
    return all_aug_train


# augmentationを行う関数
# input : 画像の配列、　output : augmentを行った後の画像の配列<br>
# augmentの種類<br>
# 回転・x軸反転・y軸反転・並進移動・ガウシアンノイズ・分散調節・拡大

def read_image(image,split_ratio=None):
    crop_size = 0.8
    insize = 227
    x_size= int(insize*(1-crop_size))
    y_size = int(insize*(1-crop_size))
    width = int(insize-2*x_size)
    height =int(insize-2*y_size)
    if args.augment:
        is_rotate = np.random.randint(0, args.rotate + 1)
        is_flip_x = np.random.randint(0, args.flip_x +1 )
        is_flip_y = np.random.randint(0, args.flip_y +1)
        is_trans =  np.random.randint(0, args.translation +1)
        is_gaussian= np.random.randint(0, args.gaussian_noise +1)
        is_variance= np.random.randint(0,args.variance+1)
        is_zoom = np.random.randint(0,args.zoom+1)

        if is_rotate: #  Transpose X and Y axis
            image =  image.transpose(0, 2, 1)
        if is_flip_x: # Flip along Y-axis
            image = np.array(cv2.flip(image, 1))
        if is_flip_y: # Flip along X-axis
            image = image[:, :, ::-1]
        if is_trans:#translation move
            M = np.float32([[1,0,random.randint(-30, 30)],[0,1,random.randint(-30, 30)]])
            image = image.transpose(1,2,0)
            image = np.array(cv2.warpAffine(image, M, (insize, insize))).transpose(2,0,1)
        if is_gaussian:
            gauss = np.random.normal(0,0.05,(3,insize,insize))
            image +=  gauss
        if is_variance:
            image /= 0.5
        if is_zoom:
            src = image.transpose(1,2,0)
            dst = src[y_size:y_size+height, x_size:x_size+width]
            image = cv2.resize(dst,(insize,insize)).transpose(2,0,1)
        return image
    else:
        return image


# 説明変数と目的変数に分ける関数
# input : (file名,label)のタプルが格納されたリスト、output : 説明変数(x_batch),目的変数(y_batch)<br>
# x_batch.shape = (sample数,channel,hight,width)

def read_xy(ff, augment):
    x_batch = []; y_batch = []
    for fl in ff:
        # fl= (filename, label)
        x_batch += [read_image(Top12_processed[fl[0]],augment)]
        y_batch += [fl[1]]
    return np.array(x_batch), np.array(y_batch, dtype=np.int32)


# this is not necessary except for in the tyt instances
os.environ["PATH"] = '/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'


# 重み更新を行う層の決定

def get_layer(args):
    layer_list = func.__dict__['_children']
    LAYER = args.output_layer
    if args.output_layer == 'fc7':
        disable = ['fc8']
    elif args.output_layer == 'fc6':
        disable = ['fc7', 'fc8']
    elif args.output_layer == 'pool5':
        disable = ['fc6', 'fc7', 'fc8']
    elif args.output_layer == 'conv10':
        disable = ['pool10']
    elif args.output_layer == 'fire9/expand3x3':
        disable = ['conv10', 'pool10']
    elif args.output_layer == 'pool8':
        disable = ['fire9', 'conv10', 'pool10']
    elif args.output_layer == 'res3d_branch2c':
        disable = layer_list[70:160]
    elif args.output_layer == 'res4f_branch2c':
        disable = layer_list[127:160]
    elif args.output_layer == 'res5c_branch2c':
        disable = layer_list[157:160]
    return disable


# ## 順伝播関数
# input : 画像配列,目的変数、　output : pred=True - 推測結果,  train=True - lossとaccuracy

def forward(im_data, y_data, train=True, pred=False):
    disable = get_layer(args)
    LAYER = args.output_layer
    im_data = xp.array(im_data, dtype=np.float32)
    im = chainer.Variable(im_data, volatile=not args.finetune)
    x, = func(inputs={'data': im}, outputs=[LAYER], disable=disable, train=train)
    if args.finetune and train:
        y = model.r1(x)
    else:
        y = model.r1(chainer.Variable(x.data, volatile=not train))
    t = chainer.Variable(xp.array(y_data), volatile=not train)    # assume that y_data is np.int32
    loss = F.softmax_cross_entropy(y, t)
    acc = F.accuracy(y, t)
    if pred:
        return y
    else:
        return loss, acc


# 精度評価関数(画像)
# 各画像に対して精度を評価するための関数<br>
# 同被写体について枚数の偏りを考慮して各個体に対して√ni/Σ√niだけの重みを付けて精度を算出

def get_loss_acc(pred, y_batch, object_val_array):
    object_val_list = list(object_val_array)
    uniq_val_list = list(np.unique(object_val_array))
    N = len(uniq_val_list)
    loss = 0
    acc = 0
    weight_list = []
    sum = 0
    for i in range(N):
        num = object_val_list.count(uniq_val_list[i])
        weight_list.append(np.sqrt(num))
        sum += np.sqrt(num)
    for i in range(N):
        y = Variable(xp.array(pred[object_val_array == uniq_val_list[i]].astype(np.float32)))
        t = Variable(xp.array(y_batch[object_val_array == uniq_val_list[i]].astype(np.int32)))
        loss += F.softmax_cross_entropy(y, t)*weight_list[i]/sum
        acc += F.accuracy(y, t)*weight_list[i]/sum
    return float(chainer.cuda.to_cpu(loss.data)) , float(chainer.cuda.to_cpu(acc.data))


# 精度評価関数(個体)

def inv_loss_acc(pred_all, y_batch_all, object_val_array):
    val_unique, num = np.unique(object_val_array, return_counts=True)
    for value in val_unique:
        bool =  object_val_array == value
        if value == val_unique[0]:
            inv_pred = np.array([np.mean(pred_all[bool], axis=0)])
            new_y_batch = np.array(np.mean(y_batch_all[bool]))
        else:
            inv_pred = np.concatenate((inv_pred, np.array([np.mean(pred_all[bool], axis =0)])))
            new_y_batch = np.append(new_y_batch,np.array(np.mean(y_batch_all[bool])))
    y = Variable(xp.array(inv_pred.astype(np.float32)))
    t = Variable(xp.array(new_y_batch.astype(np.int32)))
    loss = F.softmax_cross_entropy(y, t)
    acc = F.accuracy(y, t)
    return float(chainer.cuda.to_cpu(loss.data)) , float(chainer.cuda.to_cpu(acc.data))


# 十倍のtest time augmentを行なった際の予測

def get_pred_av(pred_all):
    for i in range(0, len(pred_all), 10):
        if i == 0:
            pred_av = np.array([np.mean(pred_all[i : i+10], axis = 0)])
        else:
            pred_av = np.concatenate((pred_av,np.array([np.mean(pred_all[i : i+10], axis = 0)])))
    return pred_av


# モデルの読み込み・生成

def get_model(args):
    if args.finetune_net == 'Alexnet':
        #func = caffe.CaffeFunction(home_dir+"/Model_data/bvlc_reference_caffenet.caffemodel")
        #f = open(home_dir + "/Model_data/caffemodel.pickle", 'wb')
        #pickle.dump(func, f)
        #f.close()
        f = open(home_dir + "/Model_data/caffemodel.pickle", 'rb')
        func = pickle.load(f)
        f.close()
    elif args.finetune_net == 'squeezenet':
        #func = caffe.CaffeFunction(home_dir+"/Model_data/squeezenet_v1.0.caffemodel")
        #f= open(home_dir + "/Model_data/squeezenet_1.0.pickle", 'wb')
        #pickle.dump(func, f)
        #f.close()
        f = open(home_dir + "/Model_data/squeezenet_1.0.pickle", 'rb')
        func = pickle.load(f)
        f.close()
    elif args.finetune_net == 'Resnet':
        #func = caffe.CaffeFunction(home_dir+"/Model_data/ResNet-50-model.caffemodel")
        #f= open(home_dir + "/Model_data/Resnet-50-model.pickle", 'wb')
        #pickle.dump(func, f)
        #f.close()
        f = open(home_dir + "/Model_data/Resnet-50-model.pickle", 'rb')
        func = pickle.load(f)
        f.close()
    if args.output_layer == 'fc7':
        output_num = 4096
    elif args.output_layer == 'fc6':
        output_num = 4096
    elif args.output_layer == 'pool5':
        output_num =9216
    elif args.output_layer == 'conv10':
        output_num = 225000
    elif args.output_layer == 'fire9/expand3x3':
        output_num = 43264
    elif args.output_layer == 'pool8':
        output_num =86528
    elif args.output_layer == 'res3d_branch2c':
        output_num = 430592
    elif args.output_layer == 'res4f_branch2c':
        output_num = 230400
    elif args.output_layer == 'res5c_branch2c':
        output_num = 131072
    n_ch = 12
    model = chainer.FunctionSet( r1 = L.Linear(output_num, n_ch))
    if args.gpu >= 0:
        model.to_gpu()
        func.to_gpu()

    return func, model


# optimizerの決定

def get_optimizer(func, model):
    #optimizer = optimizers.Adam()
    #optimizer = optimizers.RMSprop()
    optimizer = optimizers.MomentumSGD(lr = args.finetune_model_lr)
    optimizer2 = optimizers.MomentumSGD(lr = args.my_model_lr)
    optimizer.setup(model)
    optimizer2.setup(func)
    return optimizer, optimizer2


# 学習・予測

def run(all_aug_train,N,Top12_processed):
    global n_imgs_trained, sum_accuracy, sum_loss,sum_n, val_acc, val_loss, inv_acc, inv_loss
    interval_log = 1000
    n_val = 4
    n_imgs_trained = 0
    n_epoch = args.n_epoch
    sum_accuracy = 0
    sum_loss = 0
    sum_n = 0
    n_tr = 0
    batchsize = args.batchsize
    t_batchsize = args.batchsize
    if args.finetune_net =='Resnet':
        t_batchsize = 1


    N_test = len(val_list)

    for epoch in tqdm(range(n_epoch)):
        print('epoch', epoch)
        #training
        perm = np.random.permutation(N)
        for i in tqdm(range(0, N, batchsize)):
            if i % 200==0:
                print('batch', i)
            x_batch, y_batch = read_xy(all_aug_train[epoch][perm[i : i+batchsize]], augment=args.augment)
            optimizer.zero_grads()
            optimizer2.zero_grads()
            loss, acc = forward(x_batch, y_batch)
            loss.backward()
            optimizer.update()
            optimizer2.update()

            if epoch ==1 and i==0:
                write_computational_graph(loss)
            sum_loss  += float(chainer.cuda.to_cpu(loss.data)) * len(y_batch)
            sum_accuracy += float(chainer.cuda.to_cpu(acc.data)) * len(y_batch)
            sum_n += len(y_batch)
            n_imgs_trained += len(y_batch)
            if sum_n >=interval_log:
                write_log('train')
                sum_accuracy = 0
                sum_loss = 0
                sum_n = 0
                n_tr += 1
                if n_tr % n_val == 0:
                    serializers.save_npz(home_dir + '/output/' + today + '_model', model)
                    serializers.save_npz(home_dir + '/output/' + today + '_optimizer', optimizer)
                    serializers.save_npz(home_dir + '/output/' + today + '_func', func)
                    serializers.save_npz(home_dir + '/output/' + today + '_optimizer2', optimizer2)
                    # valuation
                    pred_all = None
                    y_batch_all = None
                    for ii in tqdm(range(0, N_test, t_batchsize)):
                        x_batch, y_batch = read_xy(val_list[ii : ii+t_batchsize], augment=args.augment)
                        pred_array = forward(x_batch, y_batch, train = False, pred = True).data

                        if pred_all is None:
                            pred_all = chainer.cuda.to_cpu(pred_array)
                        else:
                            pred_all = np.concatenate((pred_all, chainer.cuda.to_cpu(pred_array)))
                        if y_batch_all is None:
                            y_batch_all = y_batch
                        else:
                            y_batch_all = np.concatenate((y_batch_all, y_batch))
                    pred_av = get_pred_av(pred_all)
                    inv_loss, inv_acc = inv_loss_acc(pred_av, y_batch_all[0:len(y_batch_all):10], np.array(val_object_label))
                    val_loss, val_acc = get_loss_acc(pred_av, y_batch_all[0:len(y_batch_all):10], np.array(val_object_label))
                    sum_n += len(y_batch)

                    write_log('val')
                    write_log('inv')


# main

def main():
    global args,Top12_processed,func,model,optimizer,optimizer2,xp,train_list,val_list,home_dir, val_object_label
    home_dir = os.getcwd()
    args = arg_process(sys.argv[1:])
    train_list, val_list, val_object_label, train_object_label = get_train_val(args)
    Top12_processed = Top12_processed_data(args)
    all_aug_train = get_all_train(train_list,train_object_label)
    func, model = get_model(args)
    optimizer, optimizer2 = get_optimizer(func, model)
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        xp = cuda.cupy
    else:
        xp = np

    val_list = np.array(val_list)
    N = len(all_aug_train[0])
    all_aug_train = np.array(all_aug_train)
    log_calc(args)
    run(all_aug_train,N,Top12_processed)


if __name__ == '__main__':
    main()
