import pygame
import random
import neat
from Neat import Agents
import pickle
import os
import numpy as np

pygame.init()

#Color
grey = (100, 100, 100)
white = (255, 255, 255)
black = (0, 0, 0)


X = 600
Y = 600
display = pygame.display.set_mode((X, Y))

#Fonts
score_font = pygame.font.SysFont('Impact', 30)

#Player dimensions
player_size = 35

#Video Game Speed 
FPS = 60
clock = pygame.time.Clock()

#Distance Between Pipes
length = 100

#Event Handling
collided = pygame.USEREVENT + 1
passed = pygame.USEREVENT + 2
ai_move = pygame.USEREVENT + 3

class env:
    def __init__(self): 
        self.generation = 0

    def reset(self):
        score = 0
        pipe = pygame.Rect(X,150,100,500)
        player = pygame.Rect(150, (Y - player_size) //2, player_size, player_size)
        obstacles_down = []
        obstacles_up = []
        gates = []
        gate = pygame.Rect(X,150,100,500)
        speed = 5
        time = 0
        render = True

        return score, pipe, player, obstacles_down, obstacles_up, gates, gate, speed, time, render
    
    def observe(self, obstacles_down, player, obstacles_up, won, lost): #Data for each state in the game
        found = False
        reward = 0
        observation = []
        for pipe in obstacles_down:
            for pipe2 in obstacles_up:
                if pipe.x > player.x and found == False:
                    found = True 
                    if won:
                        reward = 1
                    if lost:
                        reward = -1
                        if player.y < 0 or player.y > Y:
                            reward -= 50
                    observation = [[pipe.x - player.x, pipe2[-1] - player.y, (pipe.y - player.y) - player_size], reward, lost]
                if found == True:
                    break
        return observation

    def draw(self, obstacles_down, obstacles_up, player, score): #Draws the Colliders
        display.fill(white)

        for pipe in obstacles_down:
            pygame.draw.rect(display, black, pipe)

        for pipe in obstacles_up:
            pygame.draw.rect(display, black, pipe)

        pygame.draw.circle(display, black, (player.x + (player_size//2), player.y + (player_size//2)), 20)
 
        score_text = score_font.render(str(score), 1, grey)

        display.blit(score_text, ((X // 2), 50))

    def runner(self, obstacles_down, obstacles_up, render, gates): #Renders the pipes
        if render == True:
            distance = 350 * len(obstacles_up)
            y_distance = random.randrange(175, 200, 1)
            height1 = random.randrange(200, 300, 1)
            height2 = height1 + y_distance
            pipe = pygame.Rect(X + distance, height2, length, Y)
            obstacles_down.append(pipe)
            pipe = pygame.Rect(X + distance, 0, length, height1)
            obstacles_up.append(pipe)
            gate = pygame.Rect(X + distance, height1, length, y_distance)
            gates.append(gate)

    def mechanics(self, player, time, obstacles_up, obstacles_down, gates): #Player collisions and game mechanics
        player.y += 3 * (time / 10)
        for pipe in obstacles_up:
            if player.colliderect(pipe):
                pygame.event.post(pygame.event.Event(collided))

        for pipe in obstacles_down:
            if player.colliderect(pipe):
                pygame.event.post(pygame.event.Event(collided))
        
        for gate in gates:
            if player.x > gate.x + length:
                gates.remove(gate)
                pygame.event.post(pygame.event.Event(passed))
        if player.y > Y or player.y < 0:
            pygame.event.post(pygame.event.Event(collided))
        
    def simulate(self, net, observation):
        action = net.activate(np.array(observation)/100)
        action = 1 if action[0] > 0.5 else 0  # Binary action: 0 or 1
        return action

    def game_mode(self, reset, file, model):
        if reset:
            with open(model, 'rb') as f:
                winner = pickle.load(f)

            # Load the NEAT config
            local_dir = os.path.dirname(__file__)
            config_path = os.path.join(local_dir, file)
            config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                        config_path)

            # Create the neural network from the winner genome
            net = neat.nn.FeedForwardNetwork.create(winner, config)
            total_reward = 0

        score, pipe, player, obstacles_down, obstacles_up, gates, gate, speed, time, render = self.reset()
        done  = False
        
        while not done:
            won = False
            lost = False

            time += 1
            clock.tick(FPS)
         
            if len(obstacles_down) > 1000:
                render = False

            for pipe in obstacles_down:
                pipe.x -= speed
                if pipe.x + length < 0:
                    obstacles_down.remove(pipe)
            
            for pipe in obstacles_up:
                pipe.x -= speed
                if pipe.x + length < 0:
                    obstacles_up.remove(pipe)
            
            for gate in gates:
                gate.x -= speed
                if gate.x + length < 0:
                    gates.remove(gate)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                
                if event.type == passed:
                    won = True
                    score += 1
                
                if event.type == collided:
                    lost = True
                    score = 0

                if reset == False:
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            if time < 8:
                                time = 8
                            height = round(50 * (15 / (time)))
                            time = 0
                            for _ in range(height):
                                player.y -= 1
                        
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if time < 8:
                            time = 8
                        height = round(50 * (15 / (time)))
                        time = 0
                        for _ in range(height):
                            player.y -= 1

            self.draw(obstacles_down, obstacles_up, player, score)
            pygame.display.update()
            self.runner(obstacles_down, obstacles_up, render, gates)
            self.mechanics(player, time, obstacles_up, obstacles_down, gates)       

            observation = self.observe(obstacles_down, player, obstacles_up, won, lost)[0]  

            if reset:
                action = self.simulate(net, observation)
                if action == 1:
                    if time < 8:
                        time = 8
                    height = round(50 * (15 / (time)))
                    time = 0
                    for _ in range(height):
                        player.y -= 1

                observation, reward, done = self.observe(obstacles_down, player, obstacles_up, won, lost)
                total_reward += reward 
    
        print(f"Score: {total_reward}")


    def ai_mode(self, genomes, config):
        self.generation += 1
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            genome.fitness = 0
            done = False
            score, pipe, player, obstacles_down, obstacles_up, gates, gate, speed, time, render = self.reset()
            self.runner(obstacles_down, obstacles_up, render, gates)
            won = False
            lost = False
            observation = self.observe(obstacles_down, player, obstacles_up, won, lost)[0]
            
            while not done:
                won = False
                lost = False
                action = self.simulate(net, observation)
                time += 1
                clock.tick(1000)
            
                if len(obstacles_down) > 1000:
                    render = False

                for pipe in obstacles_down:
                    pipe.x -= speed
                    if pipe.x + length < 0:
                        obstacles_down.remove(pipe)
                
                for pipe in obstacles_up:
                    pipe.x -= speed
                    if pipe.x + length < 0:
                        obstacles_up.remove(pipe)
                
                for gate in gates:
                    gate.x -= speed
                    if gate.x + length < 0:
                        gates.remove(gate)

                if action == 1:
                    if time < 8:
                        time = 8
                    height = round(50 * (15 / (time)))
                    time = 0
                    for _ in range(height):
                        player.y -= 1

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()

                    if event.type == passed:
                        score += 1
                        won = True
                    
                    if event.type == collided:
                        lost = True 
                
                self.draw(obstacles_down, obstacles_up, player, score)
                generation_text = score_font.render(str(self.generation), 1, grey)
                display.blit(generation_text, (25, 25))
                pygame.display.update()
                self.runner(obstacles_down, obstacles_up, render, gates)
                self.mechanics(player, time, obstacles_up, obstacles_down, gates)     
            
                observation, reward, done = self.observe(obstacles_down, player, obstacles_up, won, lost)
                genome.fitness += reward

                if done == True:
                    break

flappy = env()
#agents = Agents(100, flappy.ai_mode, 'Flappy_bird.txt', 'flappy_neat2.pkl')
#agents.train()
flappy.game_mode(True, 'Flappy_bird.txt', 'flappy_neat2.pkl')
