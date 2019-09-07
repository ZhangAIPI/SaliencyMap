import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import resnet50
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import cv2
name='deer'
cv2.namedWindow(name,cv2.WINDOW_NORMAL)
cv2.resizeWindow(name,1000,1000)
img=cv2.imread('{}.jpg'.format(name)).astype('float32')
#print(image.img_to_array(img).shape)
#img.show()
img=np.expand_dims(img,axis=0)
print(img.shape)

baseModel=resnet50.ResNet50(weights='imagenet')
#pred=baseModel.predict(img)

#n=10
#topN=resnet50.decode_predictions(pred,top=n)
#for i in topN[0]:
#   print(i)
model=tf.keras.Model(inputs=baseModel.input,outputs=baseModel.get_layer(name='fc1000').output)
def saliencyMap(img):
    with tf.GradientTape() as tape:
        Img=tf.Variable(img)
        pred=model(Img)[0]
        pred=pred[tf.argmax(pred)]

    index=np.argmax(pred)
    grad=tape.gradient(pred,Img)
    return grad[0]
def normalizationGrey(grad):
    grad=np.sum(np.abs(grad),axis=-1)
    percentile=99
    vmax=np.percentile(grad,percentile)
    vmin=np.min(grad)

    return np.clip((grad-vmin)/(vmax-vmin),0,1)
def smoothGrad(x,n=50):
    stdev=0.15*(np.max(x)-np.min(x))
    totalGrad=np.zeros(x.shape)
    for i in range(n):
        noise=np.random.normal(0,stdev,x.shape).astype('float32')
        temp=x+noise
        totalGrad+=saliencyMap(temp)
        print('iter:{}'.format(i))
    return totalGrad/n
def oneChannel(x):
    return np.sum(x,axis=2)

@tf.RegisterGradient('GuidedBP')
def ReluGP(op,grad):
        gate_g=tf.cast(grad>0,tf.float32)
        gate_y=tf.cast(op.outputs[0]>0,tf.float32)
        return grad*gate_g*gate_y
def GuidedBP(img):
    g=tf.Graph()
    with tf.GradientTape() as tape:
        with g.gradient_override_map({'Relu':'GuidedBP'}):
            Img=tf.Variable(img)
            pred=model(Img)[0]
            pred=pred[tf.argmax(pred)]
    grad=tape.gradient(pred,Img)
    return grad
def smoothGP(x,n):
    stdev=0.15*(np.max(x)-np.min(x))
    totalGrad=np.zeros(x.shape)
    for i in range(n):
        noise=np.random.normal(0,stdev,x.shape).astype('float32')
        temp=x+noise
        totalGrad+=GuidedBP(temp)
        print('iter:{}'.format(i))
    return totalGrad/n
def Integrated(x,steps=25,xBaseLine=None):
    if xBaseLine==None:
        xBaseLine=np.zeros_like(x)
    diff=x-xBaseLine
    iter=0
    grad=np.zeros_like(x)
    for alpha in np.linspace(0,1,steps):
        temp=xBaseLine+alpha*diff
        grad+=saliencyMap(temp)
    return grad*diff/steps
def smoothIntegrated(x,n,steps=25,xBaseLine=None):
    stdev=0.15*(np.max(x)-np.min(x))
    totalGrad=np.zeros(x.shape)
    for i in range(n):
        noise=np.random.normal(0,stdev,x.shape).astype('float32')
        temp=x+noise
        totalGrad+=Integrated(temp,steps,xBaseLine)
        print('iter:{}'.format(i))
    return totalGrad/n


if __name__=='__main__':
    #cv2.imshow(name,img[0].astype('uint8'))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    smooth=smoothIntegrated(img,n=50)[0]
    smooth=(normalizationGrey(smooth)*255).astype('uint8')
    plt.imsave('{}-saliencyIntegrated.png'.format(name),smooth)
    cv2.imshow(name,smooth)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



