include("strategy-basics.jl")

Base.@kwdef mutable struct OpenLoopStrategy{T1,T2,T3} <: AbstractGameStrategy
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
function (strategy::OpenLoopStrategy)(state, time)
  initial_guess =
    strategy.generate_initial_guess(strategy.player_strategies, state, time)
  warm_start_kwargs = isnothing(initial_guess) ? (;) : (; initial_guess)
  strategy.player_strategies = get_player_strategies(strategy, state, warm_start_kwargs)
  strategy.player_strategies(state, 1)
end


