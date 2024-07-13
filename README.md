# Presets

## 2 Tag Game 


```julia
game_type = "2-tag"
player_dynamics = [FAST_PLAYER, FAST_PLAYER]
initial_state_config = "random" # "random" or "static"
strategies = ["lifted", "random-1"] # "random-1" or "lifted"
strategy_config = "receding-horizon" # open-loop or receding-horizon
STRATEGY_TURN_LENGTH = 3
animation_labels = ["pursuer", "evader"]
animation_colors = [colorant"red", colorant"blue"]
```


## 3 Cooperative Tag
```julia
game_type = "3-coop-adv1"
player_dynamics = [SLOW_PLAYER, SLOW_PLAYER, FAST_PLAYER]
initial_state_config = "random" # "random" or "static"
strategies = ["lifted", "lifted", "lifted"] # "random-1" or "lifted"
strategy_config = "receding-horizon" # open-loop or receding-horizon
STRATEGY_TURN_LENGTH = 3
animation_labels = ["pursuer", "pursuer", "evader"]
animation_colors = [colorant"red", colorant"red", colorant"blue"]
```


## Herding


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