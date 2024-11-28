import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import cv2  # OpenCV를 사용하여 이미지를 로드하고 처리합니다

# 모델 로드
model = tf.keras.models.load_model('mnist_model.h5')

# 이미지 로드 및 전처리
img = cv2.imread('test_image.png', cv2.IMREAD_GRAYSCALE)  # 이미지를 흑백으로 로드
img = cv2.resize(img, (28, 28))  # 이미지를 28x28로 크기 조정
img = img.reshape((1, 28, 28, 1)).astype('float32') / 255  # 모델 입력 형식에 맞게 변환

# 예측하기
predictions = model.predict(img)

# 예측 결과 출력
predicted_class = predictions.argmax()
print(f'Predicted class: {predicted_class}')

# 이미지 시각화
plt.imshow(img.reshape(28, 28), cmap='gray')
plt.title(f'Predicted: {predicted_class}')
plt.axis('off')
plt.show()