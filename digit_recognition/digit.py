import cv2
import numpy as np

"""
Tests digit recognition model
Testfile needs to be named 'test8.png'
Predictionmodel needs to be named 'model.h5'
"""
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
    model = load_model(os.path.join(dirname, 'model.h5'))


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
model = load_model(os.path.join(dirname, 'model.h5'))

def predict_multiple(images):
"""
Predicts digits of images

Paramater:
  images List of images which should be predicted
  
Returns:
  []int Predicted numbers
"""
    if len(images) == 0:
        return []

    # prep
    inputs = []
    for img in images:
        input = cv2.resize(img, (28, 28))
        input = input.reshape(28,28,1)
        input = input.astype('float32')
        input /= 255
        inputs.append(input)

    preped = np.array(inputs)
    predictions = model.predict_on_batch(preped)

    return [ np.argmax(prediction)+1 for prediction in predictions ]


