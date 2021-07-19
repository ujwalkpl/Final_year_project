import tensorflow as tf
import pandas as pd
import numpy as np
import librosa 
import librosa.display
import cv2
import json
import os
import matplotlib.pyplot as plt
import flask
from keras.models import load_model
import logging
logging.basicConfig(level=logging.INFO)

from flask import request, jsonify,render_template
app = flask.Flask(__name__)
app.config["DEBUG"] = True

class CustomDataset(tf.keras.utils.Sequence):
    def __init__(self,imgfiles,labels,batch_size,target_size=(64,64),shuffle=False,scale=255,n_classes=1,n_channels=3):
        self.batch_size = batch_size
        self.dim        = target_size
        self.labels     = labels
        self.imgfiles   = imgfiles
        self.n_classes  = n_classes
        self.shuffle    = shuffle
        self.n_channels = n_channels
        self.scale      = scale

        self.c          = 0
        self.on_epoch_end()

    def __len__(self):
        # returns the number of batches
        return int(np.floor(len(self.imgfiles) / self.batch_size))

    def __getitem__(self, index):
        # returns one batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.imgfiles))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            img   = cv2.imread(self.imgfiles[ID])
            img   = cv2.resize(img,self.dim,interpolation = cv2.INTER_CUBIC)
            X[i,] = img / self.scale

            # Store class
            y[i] = self.labels[ID]

            self.c +=1
        return X, y #keras.utils.to_categorical(y, num_classes=self.n_classes)


class CustomPipeline(tf.keras.utils.Sequence):
    def __init__(self,data_x,data_y,batch_size=48,shuffle=False,n_classes=1):
        self.features   = data_x
        self.labels     = data_y
        self.batch_size = 48
        self.shuffle    = shuffle
        
        self.n_features = self.features.shape[1]
    
        self.n_classes  = 1
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.features) / self.batch_size))

    def __getitem__(self,index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.features))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self,indexes):
        X = np.empty((self.batch_size, self.n_features))
        y = np.empty((self.batch_size), dtype=int)

        for i, ID in enumerate(indexes):
            X[i,] = self.features[ID]
            y[i,] = self.labels[ID]
            return X, y

class MultipleInputGenerator(tf.keras.utils.Sequence):
    """Wrapper of two generatos for the combined input model"""

    def __init__(self, X1, X2, Y, batch_size,target_size=(64,64)):
        self.genX1 = CustomPipeline(X1, Y, batch_size=batch_size,shuffle=False)
        self.genX2 = CustomDataset (X2, Y, batch_size=batch_size,shuffle=False,target_size=target_size)

    def __len__(self):
        return self.genX1.__len__()

    def __getitem__(self, index):
        X1_batch, Y_batch = self.genX1.__getitem__(index)
        X2_batch, Y_batch = self.genX2.__getitem__(index)
        X_batch = [X1_batch, X2_batch]
        return X_batch, Y_batch

class TripleInputGenerator(tf.keras.utils.Sequence):
    """Wrapper of two generatos for the combined input model"""

    def __init__(self, X1, X2, X3, Y, batch_size,target_size=(64,64)):
        self.genX1 = CustomPipeline(X1, Y, batch_size=batch_size,shuffle=False)
        self.genX2 = CustomDataset (X2, Y, batch_size=batch_size,shuffle=False,target_size=target_size)
        self.genX3 = CustomPipeline(X3, Y, batch_size=batch_size,shuffle=False)
    def __len__(self):
        return self.genX1.__len__()

    def __getitem__(self, index):
        X1_batch, Y_batch = self.genX1.__getitem__(index)
        X2_batch, Y_batch = self.genX2.__getitem__(index)
        X3_batch, Y_batch = self.genX3.__getitem__(index)

        X_batch = [X1_batch, X2_batch, X3_batch]
        return X_batch, Y_batch

class SevenInputGenerator(tf.keras.utils.Sequence):
  """Wrapper of two generatos for the combined input model"""

  def __init__(self, X1, X2, X3,X4,X5,X6,X7, Y, batch_size,target_size=(64,64)):
      self.genX1 = CustomPipeline(X1, Y, batch_size=batch_size,shuffle=False)#features
      self.genX2 = CustomDataset (X2, Y, batch_size=batch_size,shuffle=False,target_size=target_size)#images
      self.genX3 = CustomPipeline(X3, Y, batch_size=batch_size,shuffle=False)#features
      self.genX4 = CustomDataset (X4, Y, batch_size=batch_size,shuffle=False,target_size=target_size)#images
      self.genX5 = CustomPipeline(X5, Y, batch_size=batch_size,shuffle=False)#features
      self.genX6 = CustomDataset (X6, Y, batch_size=batch_size,shuffle=False,target_size=target_size)#images
      self.genX7 = CustomPipeline(X7, Y, batch_size=batch_size,shuffle=False)#features
  def __len__(self):
      return self.genX1.__len__()

  def __getitem__(self, index):
      X1_batch, Y_batch = self.genX1.__getitem__(index)
      X2_batch, Y_batch = self.genX2.__getitem__(index)
      X3_batch, Y_batch = self.genX3.__getitem__(index)
      X4_batch, Y_batch = self.genX4.__getitem__(index)
      X5_batch, Y_batch = self.genX5.__getitem__(index)
      X6_batch, Y_batch = self.genX6.__getitem__(index)
      X7_batch, Y_batch = self.genX7.__getitem__(index)

      X_batch = [X1_batch, X2_batch, X3_batch,X4_batch, X5_batch, X6_batch,X7_batch]
      return X_batch, Y_batch

custpath  = ''
custpathsp= ''
custpathb=''

# def feature_extractor1(row):

#     name     = row[0]
#     custpath='custom_input'
    
#     # try:
#     audio,sr = librosa.load(row[1])
#     #For MFCCS 
#     mfccs    = librosa.feature.mfcc(y=audio,sr=sr, n_mfcc=39)
#     mfccsscaled = np.mean(mfccs.T,axis=0)
        
#     #Mel Spectogram
#     plt.axis('off') # no axis
#     plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
#     melspec  = librosa.feature.melspectrogram(y=audio,sr=sr)
#     s_db     = librosa.power_to_db(melspec, ref=np.max)
#     librosa.display.specshow(s_db)

#     savepath = os.path.join(custpath,name+'cough.png')
#     plt.savefig(savepath, bbox_inches=None, pad_inches=0)
#     plt.close()
#     # except :
#     #     print('File cannot open')
#     #     return None,None
#     return mfccsscaled,savepath


features = []
diagnoses= []
imgpaths = []
featuresb = []
imgpathsb = []
featuresp = []
imgpathsp = []

def feature_extractorb(row):

  name     = row[0]
  try:
    audio,sr = librosa.load(row[2])
    #For MFCCS 
    mfccs    = librosa.feature.mfcc(y=audio,sr=sr, n_mfcc=39)
    mfccsscaled = np.mean(mfccs.T,axis=0)
    
    #Mel Spectogram
    plt.axis('off') # no axis
    plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    melspec  = librosa.feature.melspectrogram(y=audio,sr=sr)
    s_db     = librosa.power_to_db(melspec, ref=np.max)
    librosa.display.specshow(s_db)

    savepath = os.path.join(custpathb,'-breathing'+'.png')
    plt.savefig(savepath, bbox_inches=None, pad_inches=0)
    plt.close()
  except:
    logging.info("File cannot open")  
    # print('File cannot open')
    return None,None
  return mfccsscaled,savepath



def feature_extractor(row):

  name     = row[0]
  try:
    audio,sr = librosa.load(row[1])
    #For MFCCS 
    mfccs    = librosa.feature.mfcc(y=audio,sr=sr, n_mfcc=39)
    mfccsscaled = np.mean(mfccs.T,axis=0)
    
    #Mel Spectogram
    plt.axis('off') # no axis
    plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    melspec  = librosa.feature.melspectrogram(y=audio,sr=sr)
    s_db     = librosa.power_to_db(melspec, ref=np.max)
    librosa.display.specshow(s_db)

    savepath = os.path.join(custpath,'-cough.png')
    plt.savefig(savepath, bbox_inches=None, pad_inches=0)
    plt.close()
  except:
    print('File cannot open')
    return None,None
  return mfccsscaled,savepath

def feature_extractorsp(row):

  name     = row[0]
  try:
    audio,sr = librosa.load(row[3])
    #For MFCCS 
    mfccs    = librosa.feature.mfcc(y=audio,sr=sr, n_mfcc=39)
    mfccsscaled = np.mean(mfccs.T,axis=0)
    
    #Mel Spectogram
    plt.axis('off') # no axis
    plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    melspec  = librosa.feature.melspectrogram(y=audio,sr=sr)
    s_db     = librosa.power_to_db(melspec, ref=np.max)
    librosa.display.specshow(s_db)

    savepath = os.path.join(custpathsp,'-speech'+'.png')
    plt.savefig(savepath, bbox_inches=None, pad_inches=0)
    plt.close()
  except:
    print('File cannot open')
    return None,None
  return mfccsscaled,savepath

#row='/home/sumukhmlohit/my_project_dir/virufy-covid/cough/Coswara-Data/20200413/vK2bLRNzllXNeyOMudnNSL5cfpG2/breathing-shallow.wav'
#row='/home/sumukhmlohit/my_project_dir/virufy-covid/cough/Coswara-Data/20200820/TaoyZAahOzRoDQRxb0DtDZh8Opa2/cough-shallow.wav'



#Provide inputs in the form of row ["Patient name","Audio file upload",Fever/Mild Pain,Respiratory Condition]
# @app.route('/hello', methods=['GET','POST'])
# def index():
#     value = ""
#     return render_template('project.html',value = value)

def mfcc_predictor(row_f,model):
    #print(row_f)
    mfccs,savepath  = feature_extractor(row_f)
    features.append(mfccs)
    imgpaths.append(savepath)
    diagnoses.append([row_f[4],row_f[5]])

    mfccsp,savepathp  = feature_extractorsp(row_f)
    features.append(mfccs)
    imgpaths.append(savepathp)

    mfccsb,savepathb  = feature_extractorb(row_f)
    features.append(mfccsb)
    imgpaths.append(savepathb)

    tfeaturesd = np.array([mfccs for i in range(48)])
    timgsd = np.array([savepath for i in range(48)])
    textrad = np.array([[row_f[4],row_f[5]] for i in range(48)])
    labelsd = np.array([1 for i in range(48)])

    tfeaturesdp = np.array([mfccsp for i in range(48)])
    timgsdp = np.array([savepathp for i in range(48)])

    tfeaturesdb = np.array([mfccsb for i in range(48)])
    timgsdb = np.array([savepathb for i in range(48)])


    #custom = TripleInputGenerator(tfeaturesd,timgsd,textrad,labelsd,batch_size=48,target_size=(64,64))
    logging.info(tfeaturesd,timgsd,textrad,timgsdp,tfeaturesdp,timgsdb,tfeaturesdb,labelsd,48,64)
    custom     = SevenInputGenerator(tfeaturesd,timgsd,textrad,timgsdp,tfeaturesdp,timgsdb,tfeaturesdb,labelsd,batch_size=48,target_size=(64,64))
    y_score1=model.predict_generator(custom)
    y_score1
    x=y_score1>0.5
    
    return y_score1[0][0]

# @app.route('/predict')
# def demo():
#     covid = {'result':"Negative",'value':"30"}
#     return render_template('index.html',covid = json.dumps(covid))

@app.route('/', methods=['GET','POST'])
def predict():
    value = ""
    if request.method == "GET":
        return render_template('project.html',value = value)
    if request.method == "POST":  
        input1 = request.files["breathing"]
        sym = request.form.getlist('symptoms')

        input2 = request.files["cough"]
        input3 = request.files["speech"]
        fevermp=0
        orc=0
        if 'fever' in sym or  'mp' in sym:
            fevermp=1
        if 'cld' in sym or 'st' in sym or 'cold' in sym or 'pneumonia'in sym or  'asthma' in sym:
            orc=1
        

        row_f=['RebRt0aSpjOeqnezG7i5XQn03Ql2',input2,input1,input3,fevermp,orc]
        logging.info(row_f)
        model=load_model('019--0.339--0.055---0.987.hdf5')
        result_final=mfcc_predictor(row_f,model)
        
        if result_final>=0.5:
            status="Positive"
        else:
            status="Negative"
        covid = {'result':status,'value':str(result_final*100)}
    return render_template('index.html',covid = json.dumps(covid))

app.run(debug=True)

