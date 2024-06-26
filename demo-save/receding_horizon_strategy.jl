# main actions
Base.@kwdef mutable struct RecedingHorizonStrategy{T1,T2,T3}
  solver::T1
  game::T2
  solve_kwargs::NamedTuple = (;)
  receding_horizon_strategy::Any = nothing
  time_last_updated::Int = 0
  turn_length::Int
  generate_initial_guess::T3 = (last_strategy, state, time) -> nothing
end
