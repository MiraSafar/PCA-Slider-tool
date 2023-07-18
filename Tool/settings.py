# +------------------------------+
# | PCA initialization / loading |
# +------------------------------+

NUM_PCA_COMPONENTS = 10  # Number of PCA components
PC_NAMES = ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10"]

# +-------------------------+
# | Initial PyGame Settings |
# +-------------------------+

# Screen
SCREENWIDTH = 1120
SCREENHEIGHT = 720
FPS = 60

# Colors:
# -------
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 128)
LIGHT_GRAY = (200, 200, 200)
YELLOW = (255, 255, 0)

# Layout
# ------
TITLE_DIV_HEIGHT = 80
TITLE_DIV_WIDTH = SCREENWIDTH
OUTPUT_DIV_HEIGHT = 320
OUTPUT_DIV_WIDTH = 280

# Slider settings
# ---------------
SLIDER_WIDTH = 380
SLIDER_HEIGHT = 20
HANDLE_WIDTH = 10
HANDLE_HEIGHT = 20
SLIDER_LEFT_MARGIN = 20
SLIDER_TOP_MARGIN = 32

# Files
# -----
LATENT_SPACE_FILE = 'files/latent_space_full_mnist.csv'
MODEL_FILE = 'files/conv_autoencoder_improved_mnist_20_epochs.h5'
