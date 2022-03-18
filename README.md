# Keras port of the VGGish model

Keras port of the VGGish model, originally in TFslim. 

TFslim implementation : https://github.com/tensorflow/models/tree/master/research/audioset/vggish



```python
from keras_vggish import get_vggish_model
import tensorflow as tf

weights_path = tf.keras.utils.get_file(origin="https://storage.googleapis.com/audioset/vggish_model.ckpt")

model = get_vggish_model(weights_path)

import librosa
waveform , sr = librosa.load('./test.mp3' , sr=16000)
print(model.predict( np.array([waveform]) ))


```

