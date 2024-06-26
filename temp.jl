using Plots
using BlockArrays: blocks
using LinearAlgebra: norm


function get_distance_cost_array(game::TrajectoryGame, steps::NamedTuple)
  states, inputs, strategies = steps 
  map(states, inputs) do state, input
    # game.cost.stage_cost(state, input.reference_control, [], [])[1]
    x1, x2 = blocks(state)
    sqrt(0.1 + norm(x1[1:2] - x2[1:2]))
  end
end

function get_stage_cost_array(game::TrajectoryGame, steps::NamedTuple)
  states, inputs, strategies = steps 
  map(states, inputs) do state, input
    # x1, x2 = blocks(state)
    # sqrt(0.1 + norm(x1[1:2] - x2[1:2]))
    game.cost.stage_cost(state, input.reference_control, [], [])[1]
  end
end

game_distance_costs = get_distance_cost_array(game, simulation_steps)
game_stage_costs = get_stage_cost_array(game, simulation_steps)
plot(1:length(game_stage_costs), [game_distance_costs, game_stage_costs], label=["distance" "stage"],)

