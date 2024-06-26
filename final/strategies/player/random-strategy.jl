include("strategy-basics.jl")

# taken from LiftedTrajectoryStrategy
Base.@kwdef mutable struct RandomStrategy <: AbstractPlayerStrategy
  "Player index"
  player_i::Int
  "game for bounding"
  game::TrajectoryGame
  "size of input"
  u_randomness::Int = 3
end

function limit_state_within_bounds(state, location_bound=1, speed_bound=6)
  x, y, px, py = state

  x = clamp(x, -1 * location_bound, location_bound)
  y = clamp(y, -1 * location_bound, location_bound)
  px = clamp(px, -1 * speed_bound, speed_bound)
  py = clamp(py, -1 * speed_bound, speed_bound)
  [x, y, px, py]
end


function (strategy::RandomStrategy)(state, t)
  player_state = state[Block(strategy.player_i)]

  input_u = (rand(2) .- 0.5) * 2 * strategy.u_randomness
  next_state = strategy.game.dynamics.subsystems[strategy.player_i](player_state, input_u, 1)

  # have to limit state within bounds
  next_state = limit_state_within_bounds(next_state)

  PrecomputedAction(player_state, input_u, next_state)
end
