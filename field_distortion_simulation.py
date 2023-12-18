# https://github.com/benjaminyu14
# This code simulates an animal moving between 2 food sources in the prescence of a distortion field.

# To run the code:
# python3 field_distrtion_simulation.py distortion_magnitude distortion_angle home_coord food1_coord food2_coord

# Example with distortion-vector (3, 90), home (50,50), food1 (700,280), food2 (250,580):
# python3 field_distortion_simulation.py 3 90 "(50,50)" "(700,280)" "(250,580)"


import pygame
import sys
import math
import ast

WIDTH, HEIGHT = 960, 720
FPS = 30  
WHITE = (255, 255, 255)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Animal Motion Simulation")
clock = pygame.time.Clock()

magnitude = float(sys.argv[1])
angle = float(sys.argv[2])
home1 = ast.literal_eval(sys.argv[3])
food1 = ast.literal_eval(sys.argv[4])
food2 = ast.literal_eval(sys.argv[5])

class Home(pygame.sprite.Sprite):
    def __init__(self, position):
        width = 30
        height = 30
        super().__init__()
        self.image = pygame.Surface((width, height))
        self.image.fill((255, 0, 0))
        self.rect = self.image.get_rect()
        self.rect.center = position
    
    def coords(self) -> tuple:
        return self.rect.center

class Animal(pygame.sprite.Sprite):
    def __init__(self, home_position):
        super().__init__()
        self.original_image = pygame.Surface((30, 20), pygame.SRCALPHA)
        pygame.draw.ellipse(self.original_image, (0, 128, 255), (0, 0, 30, 20))
        pygame.draw.circle(self.original_image, (255, 255, 255), (30, 10), 7)
        self.image = self.original_image.copy()
        self.rect = self.image.get_rect()
        self.rect.center = home_position
        self.speed = 0 # animal speed
        self.orientation = 0  
        self.target_food_source = None 
        self.home_position = home_position 
        self.is_at_home = True  
        self.mode_go_home = False
    
      

    def update(self, food_sources, home):
        
        def base_vector(magnitude, getting_food):
            self.speed = magnitude
            if getting_food:
                dx = self.target_food_source.rect.centerx - self.rect.centerx
                dy = self.target_food_source.rect.centery - self.rect.centery
            else:
                dx = self.home_position[0] - self.rect.centerx
                dy = self.home_position[1] - self.rect.centery
            target_orientation = math.degrees(math.atan2(dy, dx))
            self.orientation = target_orientation

        def distortion_vector(magnitude, angle, getting_food):
            rad_angle = math.radians(angle)
            x = magnitude * math.cos(rad_angle)
            y = -1 * magnitude * math.sin(rad_angle)
            if getting_food:
                distance_x = (self.target_food_source.rect.centerx - self.rect.centerx)
                distance_y = self.target_food_source.rect.centery - self.rect.centery 
            else:
                distance_x = self.home_position[0] - self.rect.centerx
                distance_y = self.home_position[1] - self.rect.centery
            distance = math.sqrt(distance_x**2 + distance_y**2) 

            return x * min(1, distance / 450), y * min(1, distance / 450)

        def moveToSource():

            base_vector(5, True)
            distort_x, distort_y = distortion_vector(magnitude, angle, True)
            rad_angle = math.radians(self.orientation)
            original_dx = self.speed * math.cos(rad_angle)
            original_dy = self.speed * math.sin(rad_angle)
            self.rect.x += original_dx + distort_x
            self.rect.y += original_dy + distort_y

            # clamp to stay in window
            self.rect.x = max(0, min(self.rect.x, WIDTH - self.rect.width))
            self.rect.y = max(0, min(self.rect.y, HEIGHT - self.rect.height))

            # rotates animal to face correct orientation
            self.image = pygame.transform.rotate(self.original_image, -self.orientation)

        def moveToHome():
            
            base_vector(5, False)
            distort_x, distort_y = distortion_vector(magnitude, angle, False)
            rad_angle = math.radians(self.orientation)
            original_dx = self.speed * math.cos(rad_angle)
            original_dy = self.speed * math.sin(rad_angle)
            self.rect.x += original_dx + distort_x
            self.rect.y += original_dy + distort_y

            # clamp to stay in window
            self.rect.x = max(0, min(self.rect.x, WIDTH - self.rect.width))
            self.rect.y = max(0, min(self.rect.y, HEIGHT - self.rect.height))

            # rotates animal to face correct orientation
            self.image = pygame.transform.rotate(self.original_image, -self.orientation)

            if pygame.sprite.collide_rect(self, home):
                if food_sources:
                    self.mode_go_home = False
                    # Set the next target_food_source (if available)
                    next_source_index = (food_sources.sprites().index(self.target_food_source) + 1) % len(food_sources.sprites())
                    self.target_food_source = food_sources.sprites()[next_source_index]
                else:
                    # All food sources visited, return home
                    self.target_food_source = None
        
        if self.is_at_home:
            # If at home, choose the next food source as the target
            if self.target_food_source == None:
                self.target_food_source = food_sources.sprites()[0]
                self.is_at_home = False

        # If the animal has a target_food_source, move towards it
        if self.target_food_source and not self.mode_go_home:

            moveToSource()
           
            # Check if the animal has reached the target_food_source
            if pygame.sprite.collide_rect(self, self.target_food_source):
                # Handle interaction with the food source (e.g., print a message)
                self.target_food_source.handle_interaction()
                print("COLLIDED")
                self.mode_go_home = True
        else:
            moveToHome()

class FoodSource(pygame.sprite.Sprite):
    def __init__(self, position):
        super().__init__()
        self.image = pygame.Surface((30, 30))
        self.image.fill((0, 255, 0))
        self.rect = self.image.get_rect()
        self.rect.center = position

    def handle_interaction(self):
        print("Animal interacts with food source at", self.rect.center)

def main():
    home = Home(home1)
    animal = Animal(home.coords())

    source_1 = FoodSource(food1)
    source_2 = FoodSource(food2)
    food_sources = pygame.sprite.Group(source_1, source_2)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # moves the animal between home and food sources
        animal.update(food_sources, home)
        
        screen.fill((0, 0, 0)) 

        # draws everything
        for food in food_sources:
            screen.blit(food.image, food.rect)

        screen.blit(animal.image, animal.rect)
        screen.blit(home.image, home.rect)

        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()
