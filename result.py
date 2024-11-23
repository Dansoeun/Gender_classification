import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, models
from sklearn.utils import shuffle
import cv2
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


df=pd.read_csv("/raid/coss35/nahyun/study/dataset/ageutk_full.csv")
df.loc[(df.age<=49) & (df.age>=20), ['files','age','gender']]

data=df['files']
target=df['gender']

xtrain = []

#기본 파일 경로 가져오기 
base_path = "/raid/coss35/nahyun/study/dataset/UTKFace/UTKFace/UTKFace/"

for i in range(len(data)):
    image_path = base_path + data.iloc[i]
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # 컬러 이미지로 읽기
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR에서 RGB로 변환
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0  # 정규화
    xtrain.append(image)

x_img_np = np.array(xtrain)
target=target.to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x_img_np, target, test_size=0.3, stratify=target)

#######Model 
model = models.Sequential()


model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224,224,3)))
model.add(layers.MaxPooling2D((2, 2)))


model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))


model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification output


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

generator=ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

hist=model.fit_generator(generator.flow(x_train,y_train,batch_size=128),epochs=5,validation_data=(x_test,y_test))

model.save("CNN_Augmentation_model.h5")

plt.figure(figsize=(18,6))

plt.subplot(1,2,1)
plt.plot(hist.history['loss'], color="blue", label = "Loss")
plt.plot(hist.history['val_loss'], color="orange", label = "Validation Loss")
plt.ylabel("Loss")
plt.xlabel("Number of Epochs")
plt.legend()

plt.subplot(1,2,2)
plt.plot(hist.history['accuracy'], color="green", label = "Accuracy")
plt.plot(hist.history['val_accuracy'], color="red", label = "Validation Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Number of Epochs")
plt.legend()

plt.show()

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)

import cv2
import numpy as np

# 이미지 로드 
image_path="/raid/coss35/nahyun/study/dataset/test4.png"
image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # 컬러 이미지로 읽기
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR에서 RGB로 변환
# 이미지 크기 확인 및 리사이징

target_size = (224, 224)  # 모델에서 기대하는 입력 크기
image_resized = cv2.resize(image, target_size)


# 정규화 (픽셀 값을 0~1 사이로 스케일링)
image_resized = image_resized.astype(np.float32) / 255.0

# 배치 차원 추가 (1, 224, 224, 3)
image_batch = np.expand_dims(image_resized, axis=0)

# Make predictions on the test set
predictions = model.predict(image_batch)
print(predictions)
# 0 : man, 1 : female
# For binary classification, you can use a threshold of 0.5
pred_classes = (predictions > 0.5).astype(int).flatten()  # Flatten to 1D array

if (pred_classes==1):
    print("여자")
else:
    print("남자")
print(pred_classes)