import os
import cv2
import numpy as np
from sklearn.model_selecategoryion import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler

classes = ['Cat','Dog']

def read_image():
    i=0
    image_features=[]
    label=[]
    for category in classes:
        for img in os.listdir(category):
            cim = os.path.join(category,img)
            img = cv2.imread(cim, cv2.IMREAD_GRAYSCALE)
            img=cv2.resize(img,(64,64))
            hog_features = hog(img, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=False)
            image_features.append(hog_features)
            label.append(category)
    return np.array(image_features),np.array(label)

images,labels = read_image()

xtrain,xtest,ytrain,ytest = train_test_split(images,labels,test_size=0.2)

classifier = SVC(kernel='rbf', C=1, gamma='scale')
classifier.fit(xtrain,ytrain)

test_image = cv2.imread("Cat/1.jpg", cv2.IMREAD_GRAYSCALE)
resized_test_image=cv2.resize(test_image,(64,64))

#Give an image path here
hog_features = hog(resized_test_image, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=False)
print(f"The image is of a :{classifier.predicategory(np.array([hog_features]))[0]}")
cv2.imshow("image",test_image)
cv2.waitKey(7000)