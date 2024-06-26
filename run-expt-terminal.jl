using BlockArrays: mortar
using GLMakie
include("expt/three_player_herd_game.jl");
game = three_player_herd_meta_game();
include("expt/lifted_trajectory_game_solver.jl");
planning_horizon = 20;
n_actions = [2 for _ in 1:num_players(game)];
solver = LiftedTrajectoryGameSolver(game, planning_horizon; n_actions);
include("expt/solve_trajectory_game.jl");
initial_state = (mortar([rand(4) for _ in 1:num_players(game)]) .- 0.5) * 4
strategy = solve_trajectory_game!(solver,game,initial_state);
time = 1;
controls = strategy(initial_state, time);
include("expt/receding_horizon_strategy.jl");
receding_horizon_strategy = RecedingHorizonStrategy(; solver, game, turn_length=5);
include("expt/simulation.jl");
number_of_simulation_steps = 2400;
simulation_steps = rollout(game.dynamics,receding_horizon_strategy,initial_state,number_of_simulation_steps;get_info=(strategy, x, t) -> strategy.receding_horizon_strategy);
include("expt/animation.jl");
GLMakie.activate!()
animate_sim_steps(game, simulation_steps; live=false, framerate=120, show_turn=true, filename="sim_steps_term_expt", show_costs=true, show_legend=true)
