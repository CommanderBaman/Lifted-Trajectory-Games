using BlockArrays: mortar
using GLMakie


include("two_player_tag_game.jl");
# game contains three main things: 
# 1. dyanmics of each player
# 2. cost of each player 
# 3. environment in which the game exists
game = two_player_meta_tag();


include("lifted_trajectory_game_solver.jl");
planning_horizon = 20;
n_actions = [2 for _ in 1:num_players(game)];

# define the solver
# the solver contains two main things:
# 1. the neural network for generating trajectories = trajectory_reference_generators
# 2. trajectory game to be solved and its solver    = trajectory_optimizers = (problem, solver)
solver = LiftedTrajectoryGameSolver(game, planning_horizon; n_actions);


include("solve_trajectory_game.jl");

# get a random initial state for the two players
# there is some scaling done here
# this refers to the x, y, px, py of the each player 
# limits between -2 to 2
initial_state = (mortar([rand(4) for _ in 1:num_players(game)]) .- 0.5) * 4

# final boss
# returns the strategies of all players in JointStrategy object
strategy = solve_trajectory_game!(
  solver, # solver  
  game, # game 
  initial_state # initial_state
);
# starting time
time = 1;
# initial call 
controls = strategy(initial_state, time);

include("receding_horizon_strategy.jl");

# construct a strategy
receding_horizon_strategy = RecedingHorizonStrategy(; solver, game, turn_length=5);

include("simulation.jl");
number_of_simulation_steps = 2400;

# do the simulation
simulation_steps = rollout(
  game.dynamics,  # dynamics
  receding_horizon_strategy, # strategy 
  initial_state, # x1
  number_of_simulation_steps;  # T
  get_info=(strategy, x, t) -> strategy.receding_horizon_strategy # get_info
);


include("animation.jl");
GLMakie.activate!()

# plot the simulation steps into a video
animate_sim_steps(
  game, simulation_steps; 
  live=false, framerate=60, show_turn=true, 
  filename="sim_steps_random_expt", 
  show_costs=true, show_legend=true
)
