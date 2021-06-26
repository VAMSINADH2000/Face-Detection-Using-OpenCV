from tensorflow.keras.models import load_model
import cv2
import numpy as np
import cvlib as cv


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

model = load_model('face_mask_detector.model')

from skimage.transform import resize

source=cv2.VideoCapture(0)

labels_dict={0:'MASK',1:'NO MASK'}
color_dict={0:(0,255,0),1:(0,0,255)}

while(True):
    ret,frame=source.read()
    # frame=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces, confidences = cv.detect_face(frame)

    for (x,y,w,h) in faces:
        (startX, startY) = x, y
        (endX, endY) = w, h
        face_img=frame[y:y+w,x:x+w]
        resized=resize(face_img,(224,224,3))
        # normalized=resized/255.0
        reshaped=np.reshape(resized,(1,224,224,3))
        result=model.predict(reshaped)

        label=np.argmax(result,axis=1)[0]
        cv2.rectangle(frame, (startX, startY), (endX, endY), color_dict[label],2)
        # cv2.rectangle(frame, (startX, startY), (endX, endY), color_dict[label],-1)
        # cv2.rectangle(frame,(x,y),(x+w,y+h),color_dict[label],2)
        # cv2.rectangle(frame,(x,y-40),(x+w,y),color_dict[label],-1)
        print(labels_dict[label],'__',label)
        cv2.putText(frame, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,1,color_dict[label],2)
		# cv2.rectangle(frame, (startX, startY), (endX, endY), color_dict[label],2)
        
    cv2.imshow('LIVE_STREAM',frame)
    key=cv2.waitKey(1)
    
    if(key==27):
        break
        
cv2.destroyAllWindows()
source.release()




