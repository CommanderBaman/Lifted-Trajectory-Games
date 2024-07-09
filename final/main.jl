println("Initializing Program...")

# outside imports 
using Colors
using GLMakie


# constant variables 
FAST_PLAYER = [5, 10]
SLOW_PLAYER = [2.5, 5]
ULTRA_SLOW_PLAYER = [1, 2]
CHOSEN_PLAYER = [2, 4]
DEFAULT_NUM_ACTIONS = 2
PLANNING_HORIZON = 20
NUM_SIMULATION_STEPS = 600
ANIMATION_FRAME_RATE = 60
ANIMATION_FILE_NAME_PREFIX = "game-video"
SHOW_COSTS = false

# global variables
game_type = "3-coop-herd"
player_dynamics = [FAST_PLAYER, FAST_PLAYER, CHOSEN_PLAYER, CHOSEN_PLAYER, CHOSEN_PLAYER]
initial_state_config = "random" # "random" or "static"
strategies = ["lifted", "lifted", "lifted", "lifted", "lifted"]
animation_labels = ["pursuer", "pursuer", "evader", "evader", "evader"]
animation_colors = [colorant"red", colorant"red", colorant"blue", colorant"blue", colorant"blue"]

println("Checking Arguments...")

# for safe herding
if contains(game_type, "herd")
  let 
    num_players = parse(Int, split(game_type, "-")[1])
    if contains(game_type, "coop")
      num_players += 2
    end
    @assert num_players == length(player_dynamics) 
    @assert num_players == length(strategies) 
    @assert num_players == length(animation_labels) 
    @assert num_players == length(animation_colors) 
  end 
end

println("Arguments are safe\nLoading Dependencies...")

# imports 
include("game/game-chooser.jl")
include("utils/main-util.jl")
include("solver/lifted-solver.jl");
include("strategies/game/receding-horizon.jl")
include("simulation/simulation.jl");
include("simulation/animation.jl");


# message
println("Dependencies Loaded\nStarting Program...")

# get game
game = form_game(game_type, player_dynamics)

# get initial state
initial_state = get_initial_state(num_players(game), initial_state_config)

# get solver 
n_actions = [DEFAULT_NUM_ACTIONS for _ in 1:num_players(game)];
lifted_solver = define_lifted_trajectory_solver(game, PLANNING_HORIZON; n_actions);

# get game strategy
receding_horizon_strategy = RecedingHorizonStrategy(; lifted_solver, game, turn_length=5, player_strategy_options=strategies);

# # simulate and animate
simulation_steps = rollout(
  game.dynamics,
  receding_horizon_strategy,
  initial_state,
  NUM_SIMULATION_STEPS;
  get_info=(strategy, x, t) -> strategy.player_strategies
);

println("Simulation done\nAnimating...")
GLMakie.activate!()
animate_sim_steps(
  game,
  simulation_steps;
  live=false,
  framerate=ANIMATION_FRAME_RATE,
  show_turn=true,
  filename=get_video_name(game_type, ANIMATION_FILE_NAME_PREFIX),
  show_costs=SHOW_COSTS,
  show_legend=true,
  player_colors=animation_colors,
  player_names=animation_labels,
)

println("$(get_video_name(game_type, ANIMATION_FILE_NAME_PREFIX)).mp4 saved")
println("Program completed successfully!")
println("Thank You!")
