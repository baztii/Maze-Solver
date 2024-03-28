#Import all libreries we need:

import pygame as py
import numpy  as np
import time
import random
import sys
import copy
import os

class AGENT:
    def __init__(self, laberynth, start, goal, iterations = 50, alpha = 0.5, gamma = 0.8, draw = None, sleep = 0, mapa = None, fill = None, which_lab = None):
        self.lab   = laberynth

        self.rows  = len(self.lab)
        self.cols  = len(self.lab[0])

        self.mapa = mapa

        self.start = start
        self.goal  = goal

        self.fill = fill

        self.which_lab = which_lab

        self.iter  = iterations

        self.draw = draw
        self.sleep = sleep
        if self.draw:
            self.draw_object = DRAW()

        self.alpha = alpha
        self.gamma = gamma

        self.actions = [(-1,0), (1,0), (0,-1), (0,1)] # up, down, left, right

        self.q_matrix = np.zeros((self.rows, self.cols, 4))

        self.pos = None
    
    def random_pos(self): # Returns a random valid position (tuple)
        possible_init_pos = [random.randint(1, self.rows - 2), random.randint(1, self.cols - 2)]

        while self.lab[possible_init_pos[0]][possible_init_pos[1]] <= -1:
            for event in py.event.get():
                if event.type == py.QUIT or event.type == py.KEYDOWN and event.key == py.K_ESCAPE:
                    sys.exit()

            possible_init_pos = [random.randint(1, self.rows - 2), random.randint(1, self.cols - 2)]
        
        return possible_init_pos

    def get_valid_action(self):
        valid_actions = []
        for index, (x, y) in enumerate(self.actions):
            new_x = self.pos[0] + x
            new_y = self.pos[1] + y

            if new_x < 0 or new_y < 0 or new_x >= len(self.lab) or new_y >= len(self.lab[0]):
                continue
       
            if self.lab[new_x][new_y] > -1:
                valid_actions.append((x, y, index))
        
        return valid_actions
    
    def take_action(self):
        actions = self.get_valid_action()
        if not actions: return "break"

        act_x, act_y, action_index = random.choice(actions)
        new_state_x  = self.pos[0] + act_x 
        new_state_y  = self.pos[1] + act_y

        instant_reward   = self.lab[new_state_x][new_state_y]
        future_reward    = max(self.q_matrix[new_state_x][new_state_y][:])
        current_q_value  = self.q_matrix[self.pos[0]][self.pos[1]][action_index]

        new_q_value = current_q_value + self.alpha * (instant_reward + self.gamma * future_reward - current_q_value)

        self.q_matrix[self.pos[0]][self.pos[1]][action_index] = new_q_value

        self.pos[0] += act_x
        self.pos[1] += act_y
    
    def iteration(self):
        t=0
        self.pos = self.random_pos()

        while self.pos != self.goal and t <= 100_000:
            for event in py.event.get():
                if event.type == py.QUIT or event.type == py.KEYDOWN and event.key == py.K_ESCAPE:
                    sys.exit()

            stop = self.take_action()

            if self.draw:
                self.__draw()

            if stop: break
            t += 1

    def __draw(self):

        self.draw_object.draw_objects()
        self.draw_object.draw_grid()
        self.draw_object.draw_menu()
        self.draw_object.transform_map_to_images(self.pos, self.draw)

        py.display.update()
        time.sleep(self.sleep)

    def train_agent(self):
        draw = DRAW()
        for iter in range(self.iter):
            draw.draw_iteration(f"Iteració {iter} de {self.iter}", self.mapa, self.fill, self.which_lab)
            self.iteration()
    
    def deploy_agent(self):
        self.pos = self.start
        positions = []
        positions.append(copy.deepcopy(self.pos))

        while self.pos != self.goal:
            for event in py.event.get():
                if event.type == py.QUIT or event.type == py.KEYDOWN and event.key == py.K_ESCAPE:
                    sys.exit()


            action = np.argmax(self.q_matrix[self.pos[0]][self.pos[1]][:])

            self.pos[0] += self.actions[action][0]
            self.pos[1] += self.actions[action][1]

            positions.append(copy.deepcopy(self.pos))

            print(self.pos)

            # Movements
        
        return positions

class DRAW:
    def __init__(self, width = 700, height = 700, block_size = 20):
        py.init()
        py.font.init()

        py.display.set_caption("Creador de laberints")

        self.import_images()

        self.width  = width
        self.height = height

        self.horitzontal_blocks = int(width  / block_size)
        self.vertical_blocks    = int(height / block_size - 10)

        self.block_size = block_size

        self.objects = {
            "Block" : -1,
            "Goal"  : 100_000,
            "Path"  : 0,
            "Start" : 0,
            -2      : -2
        }

        self.selected_object = None

        self.map = [[-2 for _ in range(self.horitzontal_blocks)]for _ in range(self.vertical_blocks)]

        self.big_screen    = py.display.set_mode((self.width, self.height))
        self.lab_rect      = py.Rect(0, 0, self.width, self.block_size * self.vertical_blocks)
        self.menu_rect     = py.Rect(0, self.block_size * self.vertical_blocks, self.width, self.height - self.block_size * self.vertical_blocks)
        self.elements_rect = [py.Rect(20 + 40 * x, self.block_size * self.vertical_blocks + 20, 20, 20) for x in range(len(self.sprites) - 3)]
        self.button_rect   = py.Rect(10, 600, 220, 78)

    def draw_objects(self):
        #To see the sizes:

        py.draw.rect(self.big_screen,"orange", self.menu_rect)
        py.draw.rect(self.big_screen,"blue", self.lab_rect)
        
    def draw_grid(self,activate = True):
        if not activate: return

        for x in range(self.horitzontal_blocks):
            py.draw.line(self.big_screen,"white", (self.block_size * x, 0), (self.block_size * x, self.block_size * self.vertical_blocks))
        
        for y in range(self.vertical_blocks):
            py.draw.line(self.big_screen,"white", (0, self.block_size * y), (self.width, self.block_size * y))

    def import_images(self):
        self.sprites = {
            "Block"  : py.image.load("Sprites/Block.png"),
            "Goal"   : py.image.load("Sprites/Goal.png"),
            "Path"   : py.image.load("Sprites/Path.png"),
            "Start"  : py.image.load("Sprites/Start.png"),
            "Player" : py.image.load("Sprites/Player.png"),
            "Button_controls" : py.image.load("Sprites/Button_controls.png"),
            "Button_tornar"   : py.image.load("Sprites/Button_tornar.png")
        }

    def mouse_click(self, mouse_pos, value = -2, fill = False):

        print(mouse_pos)

        y = mouse_pos[0] // self.block_size
        x = mouse_pos[1] // self.block_size


        if fill and value != self.map[x][y]:
            print("entering to fill")
            self.fill([(x, y)], self.map[x][y], value)

        print(f" x: {x} , y = {y} map len = {len(self.map)} map len 2 = {len(self.map[0])} ")

        self.map[x][y] = value

    def fill(self, list_of_blocks, value_to_fill, new_value):
        for x, y in list_of_blocks:
            if self.map[x][y] != value_to_fill:
                continue
        
            self.map[x][y] = new_value

            values_to_fill = []

            if x + 1 < len(self.map):
                values_to_fill.append((x + 1, y))
            
            if x - 1 >= 0:
                values_to_fill.append((x - 1, y))
            
            if y + 1 < len(self.map[0]):
                values_to_fill.append((x, y + 1))
            
            if y - 1 >= 0:
                values_to_fill.append((x, y - 1))

            self.fill(values_to_fill, value_to_fill, new_value)

    def draw_menu(self):
        for i, image in enumerate(self.sprites.values()):
            if i == 4: break
            if self.selected_object == list(self.sprites.keys())[i]:

                py.draw.rect(self.big_screen, "yellow", py.Rect(15 + 40 * i, self.block_size * self.vertical_blocks + 15, 30, 30))

            self.big_screen.blit(image, self.elements_rect[i])
        
        self.big_screen.blit(self.sprites["Button_controls"], self.button_rect)
          
    def select_object(self, mouse_pos):

        if self.button_rect.collidepoint(mouse_pos):
            self.options()

        for i, el in enumerate(self.sprites):
            if i == 4 or i == 5 or i == 6: continue
            if self.elements_rect[i].collidepoint(mouse_pos):
                self.selected_object = el
                break
        else:
            self.selected_object = None

        print(self.selected_object)

    def options(self):
        self.big_screen.fill("orange")
        rect_retrun = py.Rect(10, 600 , 197,78 )
        self.big_screen.blit(self.sprites["Button_tornar"], rect_retrun)

        loop = True

        while loop:
            loop = False
            
            for event in py.event.get():
                mouse_pos = py.mouse.get_pos()
                if event.type == py.QUIT or event.type == py.KEYDOWN and event.key == py.K_ESCAPE:
                    sys.exit()
                
                if event.type == py.MOUSEBUTTONDOWN and event.button == 1:
                    if rect_retrun.collidepoint(mouse_pos):
                        break
            else:
                loop = True


            myfont = py.font.SysFont('Comic Sans MS', 27)
            text_positions = [(5,0), (5,60), (5,90), (5,120), (5,150), (5,180), (5,210), (5,240), (5,270)]
            text_surfaces = [myfont.render(' Controls de les tecles i el ratolí:', True, "black")]
            text_surfaces.append(myfont.render(' Botó esquerra del ratolí --> Seleccionar i dibuixar', False, "black"))
            text_surfaces.append(myfont.render(' Botó dret del ratolí --> Esborrar', False, "black"))
            text_surfaces.append(myfont.render(" 'a' --> Entrenar l'agent", False, "black"))
            text_surfaces.append(myfont.render(" 'c' --> Executar l'agent", False, "black"))
            text_surfaces.append(myfont.render(" 's' --> Guardar el laberint creat", False, "black"))
            text_surfaces.append(myfont.render(" 'q' --> Activar/desactivar emplenar ", False, "black"))
            text_surfaces.append(myfont.render(" Espai --> Seleccionar el següent laberint guardat", False, "black"))
            text_surfaces.append(myfont.render(" Intro --> Obrir laberint seleccionat", False, "black"))


            for text, position in zip(text_surfaces, text_positions):
                self.big_screen.blit(text,position)




            py.display.update()

    def transform_map_to_images(self, player_pos = None, new_map = None):
        
        if new_map:
            self.map = new_map
        
        for x, row in enumerate(self.map):
            for y, num in enumerate(row):
                if num == -2: continue
                self.big_screen.blit(self.sprites[num], self.sprites[num].get_rect(topleft = (self.block_size * y, self.block_size * x)))

        if player_pos:
            self.big_screen.blit(self.sprites["Player"], self.sprites["Player"].get_rect(topleft = (self.block_size * player_pos[1], self.block_size * player_pos[0])))

    def make_laberynth(self):
        start, end = None, None
        self.laberynth = copy.deepcopy(self.map)
        for i in range(len(self.laberynth)):
            for j in range(len(self.laberynth[i])):
                if self.laberynth[i][j] == "Start":
                    start = [i, j]

                if self.laberynth[i][j] == "Goal":
                    end = [i, j]

                self.laberynth[i][j] = self.objects[self.laberynth[i][j]]

        return start,end

    def background(self):

            for x in range(100):
                for y in range(100):
                    color = (x,y,x + y)
                    rect = py.Rect(7*x,7*y,7,7)
                    py.draw.rect(self.big_screen,color,rect)

    def draw_iteration(self, iteration, mapa, fill, which_lab):
        self.map = mapa

        self.draw_objects()
        self.draw_grid()
        self.draw_menu()
        self.transform_map_to_images()
    
        myfont = py.font.SysFont('Comic Sans MS', 27)
        text_surface = myfont.render(iteration, False, "black")
        self.big_screen.blit(text_surface, (250, 510))

        myfont = py.font.SysFont('Comic Sans MS', 27)
        text_surface = myfont.render("Emplenar: ON" if fill else "Emplenar: OFF", False, "black")
        self.big_screen.blit(text_surface, (250, 600))

        text_surface = myfont.render("Laberint seleccionat: Cap" if which_lab == -1 else f"Laberint seleccionat: {which_lab}", False, "black")
        self.big_screen.blit(text_surface, (250, 640))

        py.display.update()

class CONTROL:
    def __init__(self):
        self.which_lab = -1
        self.draw = DRAW()
        self.positions = None
        self.txt_count = self.count_lab_save()
        self.imported_lab = None

    def gameloop(self):

        clock = py.time.Clock()

        fill = False
        saving = False
        time_to_saving = 0
        secs_waited = 0
        while True:
            self.txt_count = self.count_lab_save()

            if not saving:

                time_to_saving = 0
                secs_waited = 0

            if saving:
                secs_waited = time.time()

 
            for event in py.event.get():
                mouse_pos = py.mouse.get_pos()

                if event.type == py.QUIT or event.type == py.KEYDOWN and event.key == py.K_ESCAPE:
                    sys.exit()

                if py.mouse.get_pressed(num_buttons = 3)[0]:
                    if self.draw.lab_rect.collidepoint(mouse_pos) and self.draw.selected_object:
                        self.draw.mouse_click(mouse_pos, self.draw.selected_object, fill)

                if event.type == py.MOUSEBUTTONDOWN and event.button == 1:
                        if self.draw.menu_rect.collidepoint(mouse_pos):
                            self.draw.select_object(mouse_pos)

                if py.mouse.get_pressed(num_buttons = 3)[2]:
                    if self.draw.lab_rect.collidepoint(mouse_pos):
                        self.draw.mouse_click(mouse_pos, fill = fill)

                if event.type == py.KEYDOWN and event.key == py.K_a:
                    saving = False
                    time_to_saving = 0
                    secs_waited = 0

                    start, end = self.draw.make_laberynth()
                    agent = AGENT(self.draw.laberynth, start, end, mapa=self.draw.map, fill=fill, which_lab=self.which_lab)
                    agent.train_agent()
                    self.positions = agent.deploy_agent()

                if event.type == py.KEYDOWN and event.key == py.K_c:
                    if not self.positions: continue

                    for player_pos in self.positions:
                        for event in py.event.get():
                            if event.type == py.QUIT or event.type == py.KEYDOWN and event.key == py.K_ESCAPE:
                                sys.exit()
                        #self.draw.background()
                        self.draw.draw_objects()
                        self.draw.draw_grid()
                        self.draw.draw_menu()
                        self.draw.transform_map_to_images(player_pos)
                        myfont = py.font.SysFont('Comic Sans MS', 27)
                        text_surface = myfont.render("Emplenar: ON" if fill else "Emplenar: OFF", False, "black")
                        self.draw.big_screen.blit(text_surface, (250, 600))

                        text_surface = myfont.render("Laberint seleccionat: Cap" if self.which_lab == -1 else f"Laberint seleccionat: {self.which_lab}", False, "black")
                        self.draw.big_screen.blit(text_surface, (250, 640))
                        py.display.update()
                        time.sleep(.1)

                    
                    #self.draw.map = [[-2 for _ in range(self.draw.horitzontal_blocks)]for _ in range(self.draw.vertical_blocks)]

                if event.type == py.KEYDOWN and event.key == py.K_s:
                    self.save_lab()
                    saving = True

                if event.type == py.KEYDOWN and event.key == py.K_SPACE:
                    self.which_lab += 1
                    if self.which_lab == self.txt_count:
                        self.which_lab = -1
                    
                    print(self.which_lab)

                if event.type == py.KEYDOWN and event.key == py.K_q:
                    fill = not(fill)
                    print(fill)

                if event.type == py.KEYDOWN and event.key == py.K_RETURN:
                    if not self.which_lab == -1:
                        self.get_lab()

            #self.draw.background()
            self.draw.draw_objects()
            self.draw.draw_grid()
            self.draw.draw_menu()
            self.draw.transform_map_to_images(new_map=self.imported_lab)
            if saving and secs_waited != 0:
                self.saving_message(fill)
            
            myfont = py.font.SysFont('Comic Sans MS', 27)
            text_surface = myfont.render("Emplenar: ON" if fill else "Emplenar: OFF", False, "black")
            self.draw.big_screen.blit(text_surface, (250, 600))

            text_surface = myfont.render("Laberint seleccionat: Cap" if self.which_lab == -1 else f"Laberint seleccionat: {self.which_lab}", False, "black")
            self.draw.big_screen.blit(text_surface, (250, 640))


            py.display.update()

            print(time_to_saving)

            if saving and secs_waited != 0:
                time_to_saving += time.time() - secs_waited
            
            if time_to_saving >= 1:
                saving = False
                time_to_saving = 0
            
            clock.tick(120)
        
    
    def saving_message(self, fill):

        self.draw.draw_objects()
        self.draw.draw_grid()
        self.draw.draw_menu()
        self.draw.transform_map_to_images()

        myfont = py.font.SysFont('Comic Sans MS', 27)
        text_surface = myfont.render("Guardant laberint...", False, "black")
        self.draw.big_screen.blit(text_surface, (250, 510))

        myfont = py.font.SysFont('Comic Sans MS', 27)
        text_surface = myfont.render("Emplenar: ON" if fill else "Emplenar: OFF", False, "black")
        self.draw.big_screen.blit(text_surface, (250, 600))

        text_surface = myfont.render("Laberint seleccionat: Cap" if self.which_lab == -1 else f"Laberint seleccionat: {self.which_lab}", False, "black")
        self.draw.big_screen.blit(text_surface, (250, 640))

        py.display.update()
        

    def count_lab_save(self):
        list_txt = []
        list_txt = os.listdir("Laberints")
        count = 0
        for file in list_txt:
            if file.endswith('.txt'):
                count += 1
        
        return count

    def save_lab(self):
        name = f"Laberints/lab{self.txt_count}.txt"
        with open(name, "w") as f:
            for row in self.draw.map:
                for el in row:
                    f.write(f"{el},")
                
                f.write("\n")
    
    def get_lab(self):
        name = f"Laberints/lab{self.which_lab}.txt"
        with open(name, "r") as f:
            content = f.read()
            content = content.split("\n")
            for i in range(len(content)):
                content[i] = content[i].split(",")

            for i in range(len(content)):
                for j in range(len(content[i])):
                    if content[i][j] == '':
                        del content[i][j]
                    elif not content[i][j].isalpha():
                        
                        content[i][j] = int(content[i][j])
            
            del content[-1]
            
            self.draw.map = content

if __name__ == '__main__':
    control = CONTROL()
    control.gameloop()

