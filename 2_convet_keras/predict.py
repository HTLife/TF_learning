import numpy as np
import scipy.ndimage

import sys

data = scipy.ndimage.imread(sys.argv[1], flatten=True)
data = np.vectorize(lambda x: 255 - x)(np.reshape(data, (1,28,28,1)))
data = data/255.  # image value is 0.0 ~ 1.0


from keras.models import load_model
model = load_model('my_model.h5')

result = model.predict(data)
print(result)
print('Predicted number = ' + str(np.argmax(result, 1)[0]))

