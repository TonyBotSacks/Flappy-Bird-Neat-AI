# Flappy-Bird-Neat-AI
Flappy bird environment with neat-python implementation.

# Setup
  
    git clone https://github.com/TonyBotSacks/Flappy-Bird-Neat-AI.git
  
    python -m venv env
  
    env/Scripts/activate
     
    pip install -r requirements.txt 

    python "[path of Flappy Bird.py]"

The training agent with the default config file manages to play upto 200 points. Modify the config file as needed to see different results.

# Environment/Game

Flappy bird can also be played with hte keyboard controls by modifying line 297:

    flappy.game_mode(True, 'Flappy_bird.txt', 'flappy_neat.pkl')
    
to

    flappy.game_mode(False, 'Flappy_bird.txt', 'flappy_neat.pkl')

after training. 
