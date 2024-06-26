# taken from LiftedTrajectoryStrategy
Base.@kwdef mutable struct RandomStrategy{TC,TW,TI,TR} <: AbstractStrategy
  "Player index"
  player_i::Int
  "A vector of actions in continuous domain."
  trajectories::Vector{TC}
  "A collection of weights associated with each candidate action to mix over these."
  weights::TW
  "A dict-like object with additioal information about this strategy."
  info::TI
  "A random number generator to compute pseudo-random actions."
  rng::TR
  "The index of the action that has been sampled when this strategy has been querried for an \
  action the first time (needed for longer open-loop rollouts)"
  action_index::Ref{Int} = Ref(0)
  "state of the players"
  current_x::Any = []
  "size of input"
  u_randomness::Int = 3 
end

function limit_state_within_bounds(state, location_bound = 1, speed_bound = 6)
  x, y, px, py = state 

  x = clamp(x, -1 * location_bound, location_bound)
  y = clamp(y, -1 * location_bound, location_bound)
  px = clamp(px, -1 * speed_bound, speed_bound)
  py = clamp(py, -1 * speed_bound, speed_bound)
  [x,y,px,py]
end


function (strategy::RandomStrategy)(state, t)
  if t == 1
      strategy.action_index[] = sample(strategy.rng, Weights(strategy.weights))
  end

  # optimal trajectories
  (; xs, us) = strategy.trajectories[strategy.action_index[]]

  if t == 1
    strategy.current_x = xs[1]
  end

  input_u = (rand(2) .- 0.5) * 2 * strategy.u_randomness
  next_state = strategy.info.game.dynamics.subsystems[strategy.player_i](strategy.current_x, input_u, 1)

  # have to limit state within bounds
  next_state = limit_state_within_bounds(next_state)



  computed_action = PrecomputedAction(strategy.current_x, input_u, next_state)
  strategy.current_x = next_state

  computed_action
end
