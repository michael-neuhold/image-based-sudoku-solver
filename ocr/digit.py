import cv2
import numpy as np


if __name__ == '__main__':
    import os
    dirname = os.path.dirname(__file__)

    x = cv2.imread(os.path.join(dirname, 'test8.png'))
    #compute a bit-wise inversion so black becomes white and vice versa
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    # x = np.invert(x)
    #make it the right size
    x = cv2.resize(x, (28, 28))
    #convert to a 4D tensor to feed into our model
    x = x.reshape(1,28,28,1)
    x = x.astype('float32')
    x /= 255

    #perform the prediction
    from keras.models import load_model
    model = load_model(os.path.join(dirname, 'cnn.h5'))


    import time
    start = time.time()

    for _ in range (9):
        out = model.predict(x)
        print(np.argmax(out))

    end = time.time()
    print(end - start)



from keras.models import load_model
import os
dirname = os.path.dirname(__file__)
model = load_model(os.path.join(dirname, 'cnn.h5'))

def predict(digit_img):
    # bw = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
    # digit = np.invert(digit_img)
    digit = cv2.resize(digit_img, (28, 28))
    #convert to a 4D tensor to feed into our model
    digit = digit.reshape(1,28,28,1)
    digit = digit.astype('float32')
    digit /= 255

    out = model.predict(digit)

    return out


