# continuous test
println("Initializing Program...")

# outside imports 
using Colors
using GLMakie
using ProgressBars
using Suppressor

# constant variables 
FAST_PLAYER = [5, 10]
SLOW_PLAYER = [2, 4]
ULTRA_SLOW_PLAYER = [1, 2]
CHOSEN_PLAYER = [2, 4]
DEFAULT_NUM_ACTIONS = 2
PLANNING_HORIZON = 20

NUM_TRAINING_SIMULATION = 100
NUM_TRAINING_SIMULATION_STEPS = 100

NUM_SIMULATIONS = 50
NUM_SIMULATION_STEPS = 200

COST_PLOT_PREFIX = "cost-plot"
PRINT_COSTS = true

DO_ANIMATION = true
ANIMATION_FRAME_RATE = 5
ANIMATION_FILE_NAME_PREFIX = "game-video"
SHOW_COSTS = false

# global variables
game_type = "3-herd-adv"
strategy_config = "receding-horizon" # open-loop or receding-horizon
STRATEGY_TURN_LENGTH = 5
strategies = ["lifted", "lifted", "lifted"] # "random-1" or "lifted"
player_dynamics = [FAST_PLAYER, SLOW_PLAYER, SLOW_PLAYER]
initial_state_config = "random" # "random" or "static"
animation_labels = ["pursuer", "evader", "evader"]
animation_colors = [colorant"red", colorant"blue", colorant"blue"]

println("$game_type, $strategy_config-$STRATEGY_TURN_LENGTH")

# decide suffix 
is_random = any([contains(x, "random") for x in strategies])
if is_random
  FILE_SUFFIX = "$strategy_config-$STRATEGY_TURN_LENGTH-random"
else
  FILE_SUFFIX = "$strategy_config-$STRATEGY_TURN_LENGTH-lifted"
end

# checking if arguments are safe or not
@assert length(strategies) == length(player_dynamics)

# for safe herding
if contains(game_type, "herd")
  let
    num_players = parse(Int, split(game_type, "-")[1])
    @assert num_players == length(player_dynamics)
    @assert num_players == length(strategies)
    if DO_ANIMATION
      @assert num_players == length(animation_labels)
      @assert num_players == length(animation_colors)
    end
  end
end

println("Arguments are safe\nLoading Dependencies...")

# imports 
include("game/game-chooser.jl")
include("utils/costs.jl")
include("utils/reducers.jl")
include("utils/main-util.jl")
include("solver/lifted-solver.jl");
include("strategies/game/game-strategy-chooser.jl")
include("simulation/simulation.jl");
include("simulation/animation.jl");


# message
println("Dependencies Loaded\nStarting Program...")

# get game and solvers - objects that don't change over 
costs = []
game = form_game(game_type, player_dynamics);
n_actions = [DEFAULT_NUM_ACTIONS for _ in 1:num_players(game)];
lifted_solver = define_lifted_trajectory_solver(game, PLANNING_HORIZON; n_actions);

println("Training lifted solver $NUM_TRAINING_SIMULATION times...")
for i in ProgressBar(1:NUM_TRAINING_SIMULATION)
  # get initial state
  initial_state = get_initial_state(num_players(game), initial_state_config)
  # get game strategy
  strategy = form_strategy(strategy_config)
  receding_horizon_strategy = strategy(; lifted_solver, game, turn_length=STRATEGY_TURN_LENGTH, player_strategy_options=strategies)

  @suppress begin
    # simulate
    simulation_steps = rollout(
      game.dynamics,
      receding_horizon_strategy,
      initial_state,
      NUM_TRAINING_SIMULATION_STEPS;
      get_info=(strategy, x, t) -> strategy.player_strategies
    )
  end
end
println("Training done")

# disabling learning
lifted_solver.enable_learning = [false for _ in 1:num_players(game)]
timers = []

println("Simulating $NUM_SIMULATIONS times...")
for i in ProgressBar(1:NUM_SIMULATIONS)
  # get initial state
  initial_state = get_initial_state(num_players(game), initial_state_config)
  # get game strategy
  strategy = form_strategy(strategy_config)
  receding_horizon_strategy = strategy(; lifted_solver, game, turn_length=STRATEGY_TURN_LENGTH, player_strategy_options=strategies)

  
  @suppress begin
    start_time = time_ns()
    # simulate
    simulation_steps = rollout(
      game.dynamics,
      receding_horizon_strategy,
      initial_state,
      NUM_SIMULATION_STEPS;
      get_info=(strategy, x, t) -> strategy.player_strategies
    )

    states, inputs, strats = simulation_steps
    reducer = form_reducer(game_type)
    cost_for_this = get_game_cost(game, states, inputs, reducer, PLANNING_HORIZON)

    push!(costs, cost_for_this)

    end_time = time_ns()
    # convert to ms
    push!(timers, (end_time - start_time) / 1000000)

    if DO_ANIMATION & (i == NUM_SIMULATIONS)
      GLMakie.activate!()
      animate_sim_steps(
        game,
        simulation_steps;
        live=false,
        framerate=ANIMATION_FRAME_RATE,
        show_turn=true,
        filename=get_video_name(game_type, ANIMATION_FILE_NAME_PREFIX, FILE_SUFFIX),
        show_costs=SHOW_COSTS,
        show_legend=true,
        player_colors=animation_colors,
        player_names=animation_labels,
        heading=""
      )
    end
  end
end

println("Simulation done\nForming Plot...")
photo_name = get_video_name(game_type, COST_PLOT_PREFIX, FILE_SUFFIX)
# capturing in a plot
save_cost_plot(
  costs;
  heading="",
  filename=photo_name,
)

if PRINT_COSTS
  mean_cost, mean_cost_stddev = get_plot_print_values(costs)
  avg_time = sum(timers) / length(timers)
  avg_time_per_step = round(avg_time / NUM_SIMULATION_STEPS; digits=3)
  println("Random: $is_random")
  println("Strategy: $strategy_config")
  println("Cost Type: $game_type")
  println("Avg Time: $avg_time_per_step ms")
  println("Costs: $mean_cost \\pm $mean_cost_stddev")
end

println("Plot saved at $photo_name.png")


println("Program completed successfully!")
println("Thank You!")
