import pygame
import sys
from settings import *
from helpers import *
from decoder import Decoder
from ui import Layout
from tensorflow.keras.models import load_model


# +---------------------------------------+
# | Program class                         |
# |  - main class for the PCA Slider App  |
# |  - initializes and closes the program |
# |  - manipulates the program            |
# +---------------------------------------+
class Program:

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
        self.clock = pygame.time.Clock()

        self.running = True

        self.dataset = "MNIST"

        self.latent_space = np.loadtxt(LATENT_SPACE_FILE, delimiter=',')
        self.original_dataset = load_mnist_full()
        self.model = load_model(MODEL_FILE)

        self.decoder = Decoder(self.model)

        # Generate a random index from the dataset (ranges from 0 to the last index in dataset)
        self.digit_index = np.random.randint(0, len(self.latent_space) + 1)

        # pca overview is a list of PCA names as string, min and max values for each PCA and the step
        # pca_encodings is an numpy array of all digit encodings
        # to get the encodings of the right digit select it by self.pca_encodings[<<digit index>>]
        self.pca_overview, self.pca_encondings, self.pca = calculate_pca(self.latent_space)

        self.ui = Layout(self.screen, self.digit_index, self.dataset, self.pca_overview)

        # Image properties will be filled in 'select_random_digit' method
        self.decoded_image = None
        self.latent_image = None
        self.original_image = None

        self.select_random_digit()

    def run(self):
        while self.running:  # While self.running == True
            self.update()
        self.close()

    def update(self):

        # Get mouse position
        mouse_pos = pygame.mouse.get_pos()
        # Get mouse state
        mouse_buttons = pygame.mouse.get_pressed()   # It is a tuple for (Left mouse, middle mouse, right mouse buttons)

        for event in pygame.event.get():

            # Close the program when quitting:
            if event.type == pygame.QUIT:
                self.running = False

            # When mouse button is clicked
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # If "Random" button was clicked
                if self.ui.button_random.collidepoint(event.pos):
                    # Load another random digit
                    self.select_random_digit()

                # This checks if the mouse was clicked on slider rect and if yes updates the handle position
                new_pca_encoding = []

                for i, slider in enumerate(self.ui.sliders):
                    slider.check_selected(mouse_pos)
                    slider.update_handle_pos(mouse_pos, self.pca_overview[i][1], self.pca_overview[i][2])
                    new_pca_encoding.append(slider.pca_encoding)

                reconstructed_data = self.pca.inverse_transform(new_pca_encoding)

                self.latent_image = LatentSpace(reconstructed_data).image
                self.screen.blit(self.latent_image, (280, 120))

                self.decoded_image = DecodedImage(self.decoder, reconstructed_data).image
                self.screen.blit(self.decoded_image, (0, 120))

            elif event.type == pygame.MOUSEMOTION:
                # Check if mouse is pressed (indicating a drag)
                if mouse_buttons[0]:

                    new_pca_encoding = []

                    for i, slider in enumerate(self.ui.sliders):
                        slider.update_handle_pos(mouse_pos, self.pca_overview[i][1], self.pca_overview[i][2])
                        new_pca_encoding.append(slider.pca_encoding)

                    reconstructed_data = self.pca.inverse_transform(new_pca_encoding)

                    self.latent_image = LatentSpace(reconstructed_data).image
                    self.screen.blit(self.latent_image, (280, 120))

                    self.decoded_image = DecodedImage(self.decoder, reconstructed_data).image
                    self.screen.blit(self.decoded_image, (0, 120))

        # Draw sliders
        for slider in self.ui.sliders:
            slider.check_selected(mouse_pos)
            slider.draw(self.screen)

        # Update program in steady frame rate
        pygame.display.update()
        self.clock.tick(FPS)

    def select_random_digit(self):

        self.digit_index = np.random.randint(0, len(self.latent_space) + 1)

        # Render the decoded MNIST digit
        self.decoded_image = DecodedImage(self.decoder, self.latent_space[self.digit_index]).image
        self.screen.blit(self.decoded_image, (0, 120))

        # Render latent space representation of the same digit above
        self.latent_image = LatentSpace(self.latent_space, self.digit_index).image
        self.screen.blit(self.latent_image, (280, 120))

        # Render the original image
        self.original_image = OriginalImage(self.original_dataset, self.digit_index).image
        self.screen.blit(self.original_image, (560, 120))

        self.ui.update_ui(self.screen, self.digit_index, self.pca_encondings[self.digit_index])

    def close(self):
        pygame.quit()
        sys.exit()


class DecodedImage:

    def __init__(self, decoder: Decoder, latent_space: np.ndarray):

        # Load an image of a handwritten digit
        image = decoder.decode(latent_space)
        surface = pygame.surfarray.make_surface(image)
        surface_corr = pygame.transform.rotate(pygame.transform.scale(surface, (280, 280)), 270)
        self.image = pygame.transform.flip(surface_corr, True, False)


class LatentSpace:

    def __init__(self, latent_space: np.ndarray, digit_index: int = None):

        if digit_index:
            image = np.reshape(latent_space[digit_index], (6, 6))
        else:
            image = np.reshape(latent_space, (6, 6))
        surface = pygame.surfarray.make_surface(image)
        self.image = pygame.transform.rotate(pygame.transform.scale(surface, (280, 280)), 270)


class OriginalImage:

    def __init__(self, original_dataset: np.ndarray, digit_index: int):
        image = np.reshape(original_dataset[digit_index], (28, 28))
        rgb_array = np.repeat(image[..., np.newaxis], 3, axis=2)  # This changes colors to just black and white
        surface = pygame.surfarray.make_surface(rgb_array)
        surface_corr = pygame.transform.rotate(pygame.transform.scale(surface, (280, 280)), 270)
        self.image = pygame.transform.flip(surface_corr, True, False)
