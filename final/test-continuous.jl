# continuous test
println("Initializing Program...")

# outside imports 
using Colors
using GLMakie
using ProgressBars
using Suppressor

# constant variables 
FAST_PLAYER = [5, 10]
SLOW_PLAYER = [2.5, 5]
ULTRA_SLOW_PLAYER = [1, 2]
CHOSEN_PLAYER = [1, 2]
DEFAULT_NUM_ACTIONS = 2
PLANNING_HORIZON = 20
NUM_SIMULATION_STEPS = 500
NUM_SIMULATIONS = 50
NUM_TRAINING_SIMULATION = 50
NUM_TRAINING_SIMULATION_STEPS = 100
COST_PLOT_PREFIX = "cost-plot"
SHOW_COSTS = false

# global variables
game_type = "2-tag"
player_dynamics = [FAST_PLAYER, FAST_PLAYER]
initial_state_config = "random" # "random"
strategies = ["lifted", "lifted"]

# checking if arguments are safe or not
@assert length(strategies) == length(player_dynamics)

# for safe herding
if contains(game_type, "herd")
  let
    num_players = parse(Int, split(game_type, "-")[1])
    @assert num_players == length(player_dynamics)
    @assert num_players == length(strategies)
    @assert num_players == length(animation_labels)
    @assert num_players == length(animation_colors)
  end
end

println("Arguments are safe\nLoading Dependencies...")

# imports 
include("game/game-chooser.jl")
include("utils/costs.jl")
include("utils/reducers.jl")
include("utils/main-util.jl")
include("solver/lifted-solver.jl");
include("strategies/game/receding-horizon.jl")
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
  receding_horizon_strategy = RecedingHorizonStrategy(; lifted_solver, game, turn_length=5, player_strategy_options=strategies)

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

println("Simulating $NUM_SIMULATIONS times...")
for i in ProgressBar(1:NUM_SIMULATIONS)
  # get initial state
  initial_state = get_initial_state(num_players(game), initial_state_config)
  # get game strategy
  receding_horizon_strategy = RecedingHorizonStrategy(; lifted_solver, game, turn_length=5, player_strategy_options=strategies)

  @suppress begin
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
  end
end

println("Simulation done\nForming Plot...")

# capturing in a plot
save_cost_plot(
  costs;
  heading="Cost Plot",
  filename=get_video_name(game_type, COST_PLOT_PREFIX),
)

println("Plot saved at $(get_video_name(game_type, COST_PLOT_PREFIX)).png")



println("Program completed successfully!")
println("Thank You!")
