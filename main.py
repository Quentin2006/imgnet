# Example file showing a basic pygame "game loop"
import pygame

# pygame setup
pygame.init()

# dataset has 28x28 images, so we can make 1 pixel in the dataset == 25 real pixels
screen = pygame.display.set_mode((28 * 25, 28 * 25))
clock = pygame.time.Clock()
running = True
pygame.mouse.set_visible(False)
# the dataset is a 28x28 matrix holding bools (1 for white, 0 for blacks)
userVals = [[0] * 28] * 28

while running:
    mouse_x, mouse_y = pygame.mouse.get_pos()
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            print()

    # fill the screen with a color to wipe away anything from last frame
    screen.fill("black")

    # RENDER YOUR GAME HERE

    pygame.draw.circle(screen, (255, 0, 0), (mouse_x, mouse_y), 7)
    # flip() the display to put your work on screen
    pygame.display.flip()

    clock.tick(175)

pygame.quit()
