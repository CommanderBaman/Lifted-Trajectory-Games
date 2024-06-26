include("strategy-basics.jl")
include("../player/lifted-strategy.jl")
include("../player/random-strategy.jl")

Base.@kwdef mutable struct RecedingHorizonStrategy{T1,T2,T3}
  lifted_solver::T1
  game::T2
  solve_kwargs::NamedTuple = (;)
  player_strategies::Any = nothing
  player_strategy_options::Any
  time_last_updated::Int = 0
  turn_length::Int
  generate_initial_guess::T3 = (last_strategy, state, time) -> nothing
end

# get/update the player strategies
function get_player_strategies(game_strategy::RecedingHorizonStrategy, state, warm_start_kwargs)
  solver_result = game_strategy.lifted_solver(
    game_strategy.game,
    state;
    game_strategy.solve_kwargs...,
    warm_start_kwargs...,
  )
  strategies = map(
    game_strategy.player_strategy_options,
    solver_result,
    Iterators.countfrom()
  ) do strategy_option, solver, player_i

    # lifted trajectory
    if strategy_option == "lifted"
      strategy = LiftedTrajectoryStrategy(;
        player_i,
        solver.trajectories,
        solver.weights,
        info=solver.info,
        solver.rng,
      )
    
    # random trajectory
    elseif startswith(strategy_option, "random")
      strategy = RandomStrategy(;
        player_i,
        game=game_strategy.game,
        u_randomness=parse(Int, split(strategy_option, "-")[2])
      )
    end

    strategy
  end

  JointStrategy(strategies)
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


