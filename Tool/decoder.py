from classes import *
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input


# +---------------------------------------+
# | Decoder class                         |
# |  - Decoder for the selected dataset   |
# +---------------------------------------+
class Decoder:
    def __init__(self, model):
        # Only Input dimensions of the tensor itself (without the extra dimension for batches)
        decoder_input = Input(shape=(36))  # a new input tensor to be able to feed the desired layer

        # Connect the trained model layers to the new input layer (assigned via layer names)
        x = model.get_layer("dense_2")(decoder_input)
        x = model.get_layer("reshape")(x)
        x = model.get_layer("conv2d_transpose")(x)
        x = model.get_layer("conv2d_transpose_1")(x)
        x = model.get_layer("conv2d_transpose_2")(x)
        decoder_output = model.get_layer("reshape_1")(x)

        # Create the decoder
        self.decoder = Model(inputs=decoder_input, outputs=decoder_output)

    # Supply latent space array with all digits and index of the specific digit
    def decode(self, latent_space: np.ndarray):

        latent_space_reshaped = np.reshape(latent_space, (1, 36))  # The reshape is needed for the Tensorflow layer

        # Decode latent space, reshape it, multiply by 255 (normal pixel value) and round it to 2 decimal spaces
        decoded_image = np.reshape(self.decoder.predict(latent_space_reshaped), (28, 28))
        decoded_image = decoded_image * 255

        # The grayscale image obtained from the decoder is effectively converted into a 3-channel image
        # like an RGB image, where each channel is identical (which is how grayscale is represented in 3-channel space)
        final_image = np.repeat(decoded_image[..., np.newaxis], 3, axis=2)
        return final_image
