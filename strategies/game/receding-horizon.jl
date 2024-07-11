include("strategy-basics.jl")

Base.@kwdef mutable struct RecedingHorizonStrategy{T1,T2,T3} <: AbstractGameStrategy
  lifted_solver::T1
  game::T2
  solve_kwargs::NamedTuple = (;)
  player_strategies::Any = nothing
  player_strategy_options::Any
  time_last_updated::Int = 0
  turn_length::Int
  generate_initial_guess::T3 = (last_strategy, state, time) -> nothing
end

# strategy calls
function (strategy::RecedingHorizonStrategy)(state, time)
  plan_exists = !isnothing(strategy.player_strategies)
  time_along_plan = time - strategy.time_last_updated + 1
  plan_is_still_valid = 1 <= time_along_plan <= strategy.turn_length

  update_plan = !plan_exists || !plan_is_still_valid
  if update_plan
    initial_guess =
      strategy.generate_initial_guess(strategy.player_strategies, state, time)
    warm_start_kwargs = isnothing(initial_guess) ? (;) : (; initial_guess)
    strategy.player_strategies = get_player_strategies(strategy, state, warm_start_kwargs)
    strategy.time_last_updated = time
    time_along_plan = 1
  end

  strategy.player_strategies(state, time_along_plan)
end


