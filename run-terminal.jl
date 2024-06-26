using BlockArrays: mortar
using GLMakie
include("demo/two_player_tag_game.jl");
game = two_player_meta_tag();
include("demo/lifted_trajectory_game_solver.jl");
planning_horizon = 20;
n_actions = [2 for _ in 1:num_players(game)];
solver = LiftedTrajectoryGameSolver(game, planning_horizon; n_actions);
include("demo/solve_trajectory_game.jl");
initial_state = (mortar([rand(4) for _ in 1:num_players(game)]) .- 0.5) * 4
strategy = solve_trajectory_game!(solver, game, initial_state);
time = 1;
controls = strategy(initial_state, time);
include("demo/receding_horizon_strategy.jl");
receding_horizon_strategy = RecedingHorizonStrategy(; solver, game, turn_length = 5);
include("demo/simulation.jl");
number_of_simulation_steps =600; # for 10 seconds
simulation_steps = rollout(game.dynamics,receding_horizon_strategy,initial_state,number_of_simulation_steps;get_info = (strategy, x, t) -> strategy.receding_horizon_strategy);
include("demo/animation.jl");
GLMakie.activate!()
sl = true
animate_sim_steps(game,simulation_steps;live = false, framerate = 10,show_turn = true, show_legend = sl, filename = "sim_steps_term")
