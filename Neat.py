import neat
import pickle
import os

class Agents:

    def __init__(self, generations, eval_genomes, file, model):
        self.generations = generations
        self.eval_genomes = eval_genomes
        self.file = file
        self.model = model

    def run(self, config_file):
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                    config_file)
        p = neat.Population(config)
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)

        winner = p.run(self.eval_genomes, self.generations)

        # Save the winner
        with open(self.model, 'wb') as output:
            pickle.dump(winner, output, 1)

        print('\nBest genome:\n{!s}'.format(winner))
    
    def train(self):
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, self.file)
        self.run(config_path)


