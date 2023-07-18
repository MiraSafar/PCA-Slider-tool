from classes import *


class Layout:

    def __init__(self, screen, digit_index: int, dataset: str, pca_overview: list):

        # Standard layout
        if dataset == 'MNIST':

            screen.fill(WHITE)

            # Fonts & Labels
            # ------------------------
            self.font_title = pygame.font.SysFont('calibri', 36)
            self.font_small = pygame.font.SysFont('calibri', 20)
            # Title text
            self.title = self.font_title.render('MNIST handwritten digits', True, BLACK, WHITE)
            # Label for decoded image
            self.decoded_label = self.font_small.render(f"Decoded digit Index: {digit_index}", True, WHITE, BLACK)
            # Label latent space representation
            self.latent_text = self.font_small.render("Latent space representation", True, WHITE, BLACK)
            # Label for original image
            self.original_label = self.font_small.render(f"Original digit Index: {digit_index}", True, WHITE, BLACK)
            # Label for randomize button
            self.random_text = self.font_small.render("RANDOM DIGIT", True, BLACK, GREEN)

            # Title div
            # ---------
            self.title_div = pygame.Rect(0, 0, TITLE_DIV_WIDTH, TITLE_DIV_HEIGHT)
            pygame.draw.rect(screen, BLACK, self.title_div, 2)  # "2" at the end renders border width
            # Render title text
            screen.blit(self.title, (self.title_div.x + 30, self.title_div.y + 30))

            # Main output divs
            # ----------------
            # Div for image obtained from the decoder
            self.output_div_1 = pygame.Rect(0, TITLE_DIV_HEIGHT, OUTPUT_DIV_WIDTH, OUTPUT_DIV_HEIGHT)
            pygame.draw.rect(screen, BLACK, self.output_div_1)
            # Render the decoded image label
            screen.blit(self.decoded_label, (self.output_div_1.x + 15, self.output_div_1.y + 15))

            # Div for latent space representation
            self.output_div_2 = pygame.Rect(280, 80, OUTPUT_DIV_WIDTH, OUTPUT_DIV_HEIGHT)
            pygame.draw.rect(screen, BLACK, self.output_div_2)
            # Render latent space label
            screen.blit(self.latent_text, (self.output_div_2.x + 15, self.output_div_2.y + 15))

            # Div for original dataset image
            self.output_div_3 = pygame.Rect(560, 80, OUTPUT_DIV_WIDTH, OUTPUT_DIV_HEIGHT)
            pygame.draw.rect(screen, BLACK, self.output_div_3)
            screen.blit(self.original_label, (self.output_div_3.x + 15, self.output_div_3.y + 15))

            # Interface buttons (divs)
            # ------------------------
            # Button to display random datapoint
            self.button_random = pygame.Rect(840, 80, 280, 60)
            pygame.draw.rect(screen, GREEN, self.button_random)
            # Render randomize button label
            screen.blit(self.random_text, (self.button_random.x + 15, self.button_random.y + 15))

            # Button to go to a specific datapoint
            self.button_goto = pygame.Rect(840, 140, 280, 60)
            pygame.draw.rect(screen, BLACK, self.button_goto, 2)

            # Button to reset sliders back to the original values
            self.button_reset = pygame.Rect(840, 200, 280, 60)
            pygame.draw.rect(screen, BLACK, self.button_reset, 2)

            self.sliders = []
            self.pca_overview = pca_overview

    def generate_sliders(self, screen, pca_encodings: np.ndarray):
        # Instantiate sliders
        # -------------------

        # Before generating new sliders purge the original list of sliders:
        self.sliders = []

        for i in range(NUM_PCA_COMPONENTS):
            # Calculate handle positions based on pca encoding value and min and max of that pca encoding
            handle_pos = pca_to_pixels(pca_encodings[i], self.pca_overview[i][1], self.pca_overview[i][2])
            handle_pos = handle_pos + SLIDER_LEFT_MARGIN  # Add left margin to the handle position

            if i < 5:
                self.sliders.append(Slider(0 + SLIDER_LEFT_MARGIN,
                                           400 + (64 * i) + SLIDER_TOP_MARGIN,
                                           i + 1, pca_encodings[i],
                                           handle_pos))
            else:
                self.sliders.append(Slider(420 + SLIDER_LEFT_MARGIN,
                                           80 + (64 * i) + SLIDER_TOP_MARGIN, i + 1,
                                           pca_encodings[i],
                                           handle_pos + 420))

    def update_ui(self, screen, digit_index: int, pca_encodings: np.ndarray):
        self.decoded_label = self.font_small.render(f"Decoded digit Index: {digit_index}", True, WHITE, BLACK)
        self.original_label = self.font_small.render(f"Original digit Index: {digit_index}", True, WHITE, BLACK)
        screen.blit(self.decoded_label, (self.output_div_1.x + 15, self.output_div_1.y + 15))
        screen.blit(self.original_label, (self.output_div_3.x + 15, self.output_div_3.y + 15))

        self.generate_sliders(screen, pca_encodings)

        for slider in self.sliders:
            slider.draw(screen)


class Slider:

    def __init__(self, left_pos, top_pos, pca_index, pca_encoding, handle_pos):

        self.left_pos = left_pos
        self.top_pos = top_pos

        self.handle_pos = handle_pos

        self.font_pca = pygame.font.SysFont('calibri', 20)
        self.pca_index = pca_index
        self.pca_encoding = pca_encoding

        self.pca_label = None
        self.selected = False

    def draw(self, screen):
        # Draw slider rect
        container_rect = pygame.Rect(self.left_pos, self.top_pos, SLIDER_WIDTH, SLIDER_HEIGHT)
        pygame.draw.rect(screen, LIGHT_GRAY, container_rect)

        # Draw slider handle
        handle_rect = pygame.Rect(self.handle_pos, self.top_pos, HANDLE_WIDTH, HANDLE_HEIGHT)
        if self.selected:
            pygame.draw.rect(screen, YELLOW, handle_rect)
        else:
            pygame.draw.rect(screen, RED, handle_rect)

        # Draw the PCA index and number container and contents
        # ----------------------------------------------------
        # A new rect to draw over pca labels is needed to wipe the screen clean (there were residual numbers left there)
        # There should be a better solution for this overlapping problem than this
        label_rect = pygame.Rect(container_rect.x + 150, container_rect.y - 26, 380, 20)
        pygame.draw.rect(screen, WHITE, label_rect)

        self.pca_label = self.font_pca.render(
            f"PCA {self.pca_index}  -->  {round(self.pca_encoding, 2)}",
            True, BLACK, WHITE)
        screen.blit(self.pca_label, (container_rect.x + 150, container_rect.y - 26))

    def check_selected(self, mouse_pos):

        mouse_x, mouse_y = mouse_pos

        # Calculate if the mouse was clicked inside the slider rectangle
        if self.left_pos <= mouse_x <= (self.left_pos + SLIDER_WIDTH) and self.top_pos <= mouse_y <= (self.top_pos + SLIDER_HEIGHT):
            self.selected = True
        else:
            self.selected = False

    def update_handle_pos(self, mouse_pos, pca_max, pca_min):
        x, y = mouse_pos

        # Only update x position of handle (horizontal slider)
        if self.left_pos <= x <= self.left_pos + SLIDER_WIDTH - HANDLE_WIDTH and self.selected:
            self.handle_pos = x
            self.pca_encoding = pixels_to_pca(self.handle_pos - self.left_pos, pca_max, pca_min)

