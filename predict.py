
from PIL import Image
import numpy as np
import os

#批量测试，存入到文件夹中
def change_health(weight_path,open_path,save_path):
    model = get_resnet(None,None,3)
    model.load_weights(weight_path)
    for root,dirs,files in os.walk(open_path):
        print(files)
        for filespath in files:
            #model.load_weights(weight_path)
            pic = Image.open(open_path+'/'+filespath)
            shape =pic.size 
            img = np.array(pic)/127.5 - 1
            img = np.expand_dims(img,axis=0)
            fake = (model.predict(img)*0.5 + 0.5)*255
            face = Image.fromarray(np.uint8(fake[0]))
            #face.show()
            face.save(save_path+"/"+filespath)
            print('------'+filespath+' is finished------')

if __name__=='__main__':

    #----------resultsA:健康转病害----------
    from nets.resnet_ecbam import get_resnet
    #权重包路径
    weight_path = r"weights\healthy2disease\g_AB_epoch.h5"
    #测试的文件夹
    open_path = r'datasets\healthy2disease\testA'
    #保存的文件夹
    save_path = r'resultsA'
    change_health(weight_path,open_path,save_path)
    
    
    