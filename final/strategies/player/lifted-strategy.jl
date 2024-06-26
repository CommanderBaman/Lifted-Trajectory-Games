using StatsBase: Weights, sample
using BlockArrays: Block

include("strategy-basics.jl")

# lifted strategy
Base.@kwdef struct LiftedTrajectoryStrategy{TC,TW,TI,TR} <: AbstractPlayerStrategy
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
end



function (strategy::LiftedTrajectoryStrategy)(state, t)
  if t == 1
    strategy.action_index[] = sample(strategy.rng, Weights(strategy.weights))
  end

  (; xs, us) = strategy.trajectories[strategy.action_index[]]
  if xs[t] != state[Block(strategy.player_i)]
    throw(
      ArgumentError("""  
                      This strategy is only valid for states on its trajectory but has been \
                      called for an off trajectory state instead which will likely not \
                      produce meaningful results.  
                      """),
    )
  end

  PrecomputedAction(xs[t], us[t], xs[t+1])
end
