
import tensorflow as tf 
from . import yamnet_features
from . import yamnet_params
from . import vggish_params 

def waveform_to_features(waveform):
    """Creates VGGish features using the YAMNet feature extractor."""
    waveform = waveform[0] #[None]
    params = yamnet_params.Params(
        sample_rate=vggish_params.SAMPLE_RATE,
        stft_window_seconds=vggish_params.STFT_WINDOW_LENGTH_SECONDS,
        stft_hop_seconds=vggish_params.STFT_HOP_LENGTH_SECONDS,
        mel_bands=vggish_params.NUM_MEL_BINS,
        mel_min_hz=vggish_params.MEL_MIN_HZ,
        mel_max_hz=vggish_params.MEL_MAX_HZ,
        log_offset=vggish_params.LOG_OFFSET,
        patch_window_seconds=vggish_params.EXAMPLE_WINDOW_SECONDS,
        patch_hop_seconds=vggish_params.EXAMPLE_HOP_SECONDS)
    log_mel_spectrogram, features = yamnet_features.waveform_to_log_mel_spectrogram_patches(
        waveform, params)
    return features[: ,: , : , None]


def load_slim_weights(model , checkpoint_path):
    slim_vars = tf.train.load_checkpoint( checkpoint_path )

    model.get_layer('conv1').set_weights([ slim_vars.get_tensor('vggish/conv1/weights') , slim_vars.get_tensor('vggish/conv1/biases') ])
    model.get_layer('conv2').set_weights([ slim_vars.get_tensor('vggish/conv2/weights') , slim_vars.get_tensor('vggish/conv2/biases') ])

    model.get_layer('conv3_1').set_weights([ slim_vars.get_tensor('vggish/conv3/conv3_1/weights') , slim_vars.get_tensor('vggish/conv3/conv3_1/biases') ])
    model.get_layer('conv3_2').set_weights([ slim_vars.get_tensor('vggish/conv3/conv3_2/weights') , slim_vars.get_tensor('vggish/conv3/conv3_2/biases') ])


    model.get_layer('conv4_1').set_weights([ slim_vars.get_tensor('vggish/conv4/conv4_1/weights') , slim_vars.get_tensor('vggish/conv4/conv4_1/biases') ])
    model.get_layer('conv4_2').set_weights([ slim_vars.get_tensor('vggish/conv4/conv4_2/weights') , slim_vars.get_tensor('vggish/conv4/conv4_2/biases') ])


    model.get_layer('fc1_1').set_weights([ slim_vars.get_tensor('vggish/fc1/fc1_1/weights') , slim_vars.get_tensor('vggish/fc1/fc1_1/biases') ])
    model.get_layer('fc1_2').set_weights([ slim_vars.get_tensor('vggish/fc1/fc1_2/weights') , slim_vars.get_tensor('vggish/fc1/fc1_2/biases') ])
    model.get_layer('fc2').set_weights([ slim_vars.get_tensor('vggish/fc2/weights') , slim_vars.get_tensor('vggish/fc2/biases') ])



def get_vggish_model(slim_checkpoint_path=None):
    inp = tf.keras.layers.Input((None,))
    feats = tf.keras.layers.Lambda(waveform_to_features)(inp)


    x = tf.keras.layers.Conv2D( 64, (3,3) , strides=(1, 1) , activation='relu' , padding='same' , name='conv1')(feats)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same' )(x)
    x = tf.keras.layers.Conv2D( 128, (3,3) , strides=(1, 1) , activation='relu' , padding='same', name='conv2' )(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    x = tf.keras.layers.Conv2D( 256, (3,3) , strides=(1, 1) , activation='relu' , padding='same' , name="conv3_1" )(x)
    x = tf.keras.layers.Conv2D( 256, (3,3) , strides=(1, 1) , activation='relu' , padding='same' , name="conv3_2")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)


    x = tf.keras.layers.Conv2D( 512, (3,3) , strides=(1, 1) , activation='relu' , padding='same', name="conv4_1")(x)
    x = tf.keras.layers.Conv2D( 512, (3,3) , strides=(1, 1) , activation='relu' , padding='same' , name="conv4_2")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(4096 , activation='relu' , name='fc1_1' )(x)
    x = tf.keras.layers.Dense(4096 , activation='relu'  , name='fc1_2')(x)
    x = tf.keras.layers.Dense(vggish_params.EMBEDDING_SIZE  , name='fc2')(x)


    model = tf.keras.models.Model(inp , x)

    if slim_checkpoint_path is not None:
        load_slim_weights(model ,  slim_checkpoint_path )

    return model 




# def get_vggish_model(slim_checkpoint_path=None):
#     model_single = get_vggish_model_single(slim_checkpoint_path=slim_checkpoint_path)
#     inputs = tf.keras.Input(shape=(None,))
#     inputs_ = tf.keras.layers.Lambda( lambda x : x[: , None ]) (inputs) # add extra dim, becuse single model 


