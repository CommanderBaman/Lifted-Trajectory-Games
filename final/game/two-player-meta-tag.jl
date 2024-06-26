# get the basics
include("game-basics.jl")

# main function
function two_player_meta_tag(;
  n_environment_sides=5,
  environment_radius=4,
  coupling_constraints=nothing,
  control_penalty=0.1,
  # array of [state_bounds, control_bounds] for each player
  player_bounds=[[5,10], [5,10]],
  distance_metric=norm,
)
  cost = let
      function stage_cost(x, u, t, context_state)
          x1, x2 = blocks(x)
          u1, u2 = blocks(u)
          c =
              # the first two contain the actual coordinate of the person 
              sqrt(distance_metric(x1[1:2] - x2[1:2]) + 0.1) +
              control_penalty * (distance_metric(u1) - distance_metric(u2))
          [c, -c]
      end

      function reducer(scs)
          reduce(.+, scs) ./ length(scs)
      end

      TimeSeparableTrajectoryGameCost(stage_cost, reducer, ZeroSumCostStructure(), 1.0)
  end
  # both the player have the same dynamics => replicating
  dynamics = ProductDynamics([get_dynamics_from_bound(player_bounds[idx][1], player_bounds[idx][2]) for idx in 1:2] |> Tuple)
  env = PolygonEnvironment(n_environment_sides, environment_radius)
  TrajectoryGame(dynamics, cost, env, coupling_constraints)
end
