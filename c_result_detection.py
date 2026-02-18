import cv2
import numpy as np
from tensorflow.keras import models

word_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M',
    13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

model = models.load_model("data/output/model.keras")

image = cv2.imread('data/letter_n.jpg')
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# img_gray = cv2.bitwise_not(img_gray)
# cv2.imshow("grey",img_gray)


img_final = cv2.resize(img_gray, (28, 28))
img_final = img_final.astype("float32") / 255.0
img_final = img_final.reshape(1, 28, 28, 1)

pred = model.predict(img_final)
pred_class = np.argmax(pred)
pred_word = word_dict[pred_class]
print("\nPrediction index:\t", pred_class)
print("\nPredicted character\t:", pred_word)

cv2.putText(image, "Prediction: " + pred_word, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.3, color=(0, 0, 255))
cv2.imshow('Character Recognized display : ', image)

cv2.waitKey(0)
cv2.destroyAllWindows()
