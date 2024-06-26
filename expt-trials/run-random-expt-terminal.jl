





using BlockArrays: mortar
using GLMakie
include("two_player_tag_game.jl");
game = two_player_meta_tag();
include("lifted_trajectory_game_solver.jl");
planning_horizon = 20;
n_actions = [2 for _ in 1:num_players(game)];
solver = LiftedTrajectoryGameSolver(game, planning_horizon; n_actions);
initial_state = (mortar([vcat(rand(2), [0.5, 0.5]) for _ in 1:num_players(game)]) .- 0.5) * 4
time = 1;
evader_randomness = 2;
include("solve_trajectory_game_random.jl");
strategy = solve_trajectory_game!(solver,game,initial_state; evader_u_randomness = evader_randomness);
controls = strategy(initial_state, time);
include("strategy/receding_horizon_strategy.jl");
receding_horizon_strategy = RecedingHorizonStrategy(; solver, game, turn_length=5);
number_of_simulation_steps = 600;
include("simulation.jl");
simulation_steps = rollout(game.dynamics,receding_horizon_strategy,initial_state,number_of_simulation_steps;get_info=(strategy, x, t) -> strategy.receding_horizon_strategy, evader_u_randomness = evader_randomness);
include("animation.jl");
GLMakie.activate!()
animate_sim_steps(game,simulation_steps;live=false,framerate=60,show_turn=true,filename="sim_steps_random_expt",show_costs=false, show_legend=true)
