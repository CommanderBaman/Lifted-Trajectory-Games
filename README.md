# Presets

## 2 Tag Game 

### Lifted vs Lifted

```julia
# constant variables 
FAST_PLAYER = [5, 10]
SLOW_PLAYER = [2, 5]
DEFAULT_NUM_ACTIONS = 2
PLANNING_HORIZON = 20
NUM_SIMULATION_STEPS = 600
ANIMATION_FRAME_RATE = 60
ANIMATION_FILE_NAME_PREFIX = "game-video"
SHOW_COSTS = false

# global variables
game_type = "2-tag"
player_dynamics = [FAST_PLAYER, FAST_PLAYER]
initial_state_config = "random" # "random"
strategies = ["lifted", "lifted"]
animation_labels = ["pursuer", "evader"]
animation_colors = [colorant"red", colorant"blue"]
```


## 3 Players = 2 Cooperative Pursuers + 1 Fast Evader

### All lifted


```julia
# constant variables 
FAST_PLAYER = [5, 10]
SLOW_PLAYER = [2, 5]
DEFAULT_NUM_ACTIONS = 2
PLANNING_HORIZON = 20
NUM_SIMULATION_STEPS = 600
ANIMATION_FRAME_RATE = 60
ANIMATION_FILE_NAME_PREFIX = "game-video"
SHOW_COSTS = false 

# global variables
game_type = "3-coop"
player_dynamics = [SLOW_PLAYER, SLOW_PLAYER, FAST_PLAYER]
initial_state_config = "random" # "random" or "static"
strategies = ["lifted", "lifted", "lifted"]
animation_labels = ["pursuer", "pursuer", "evader"]
animation_colors = [colorant"red", colorant"red", colorant"blue"]
```



## Cooperative Herding

```julia
# global variables
game_type = "5-coop-herd"
player_dynamics = [FAST_PLAYER, FAST_PLAYER, CHOSEN_PLAYER, CHOSEN_PLAYER, CHOSEN_PLAYER, CHOSEN_PLAYER, CHOSEN_PLAYER]
initial_state_config = "random" # "random" or "static"
strategies = ["lifted", "lifted", "lifted", "lifted", "lifted", "lifted", "lifted"]
animation_labels = ["pursuer", "pursuer", "evader", "evader", "evader", "evader", "evader"]
animation_colors = [colorant"red", colorant"red", colorant"blue", colorant"blue", colorant"blue", colorant"blue", colorant"blue"]
```