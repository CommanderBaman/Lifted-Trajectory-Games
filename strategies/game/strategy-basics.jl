
include("../player/lifted-strategy.jl")
include("../player/random-strategy.jl")

abstract type AbstractGameStrategy end


# get/update the player strategies
function get_player_strategies(game_strategy::AbstractGameStrategy, state, warm_start_kwargs)
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
