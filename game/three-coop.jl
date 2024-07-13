# get the basics
include("game-basics.jl")

# TODO: can be generalized to mulitple n using vectors

# main function
function three_player_coop_basic(;
  n_environment_sides=5,
  environment_radius=4,
  coupling_constraints=nothing,
  control_penalty=0.1,
  player_bounds=[[2, 5], [2, 5], [5, 10]],
  distance_metric=norm,
)
  cost = let
    function stage_cost(x, u, t, context_state)
      x1, x2, x3 = blocks(x)
      u1, u2, u3 = blocks(u)

      # distances
      v12 = x2[1:2] - x1[1:2]
      v13 = x3[1:2] - x1[1:2]
      d12 = sqrt(distance_metric(v12) + 0.01)
      d13 = sqrt(distance_metric(v13) + 0.01)
      v23 = x3[1:2] - x2[1:2]
      d23 = sqrt(distance_metric(v23) + 0.01)

      # directly from the lifted trajectory paper
      c1 = d13 + control_penalty * (distance_metric(u1) - distance_metric(u3))

      c2 = d23 + control_penalty * (distance_metric(u2) - distance_metric(u3))

      # evader should only depend on distances from pursuer
      c3 = -1 * (c1 + c2)

      [c1, c2, c3]
    end

    function reducer(scs)
      reduce(.+, scs) ./ length(scs)
    end

    TimeSeparableTrajectoryGameCost(stage_cost, reducer, GeneralSumCostStructure(), 1.0)
  end
  dynamics = ProductDynamics([get_dynamics_from_bound(bound[1], bound[2]) for bound in player_bounds] |> Tuple)
  env = PolygonEnvironment(n_environment_sides, environment_radius)
  TrajectoryGame(dynamics, cost, env, coupling_constraints)
end

# takes in account distance between pursuers
function three_player_coop_adv1(;
  n_environment_sides=5,
  environment_radius=4,
  coupling_constraints=nothing,
  control_penalty=0.1,
  player_bounds=[[2, 5], [2, 5], [5, 10]],
  distance_metric=norm,
)
  cost = let
    function stage_cost(x, u, t, context_state)
      x1, x2, x3 = blocks(x)
      u1, u2, u3 = blocks(u)

      # distances
      v12 = x2[1:2] - x1[1:2]
      v13 = x3[1:2] - x1[1:2]
      v23 = x3[1:2] - x2[1:2]
      d12 = sqrt(distance_metric(v12) + 0.01)
      d13 = sqrt(distance_metric(v13) + 0.01)
      d23 = sqrt(distance_metric(v23) + 0.01)

      # directly from the lifted trajectory paper
      c1 = d13 + control_penalty * (distance_metric(u1) - distance_metric(u3))

      c2 = d23 + control_penalty * (distance_metric(u2) - distance_metric(u3))

      # evader should only depend on distances from pursuer
      c3 = -1 * (c1 + c2)

      # to promote distance between them for diverse area coverage
      # more distance between them => more area covered 
      # thinking of using 1/d12, so that higher distances are not supported
      c1 -= 0.5 * d12
      c2 -= 0.5 * d12

      [c1, c2, c3]
    end

    function reducer(scs)
      reduce(.+, scs) ./ length(scs)
    end

    TimeSeparableTrajectoryGameCost(stage_cost, reducer, GeneralSumCostStructure(), 1.0)
  end
  dynamics = ProductDynamics([get_dynamics_from_bound(bound[1], bound[2]) for bound in player_bounds] |> Tuple)
  env = PolygonEnvironment(n_environment_sides, environment_radius)
  TrajectoryGame(dynamics, cost, env, coupling_constraints)
end



# takes in account distance between pursuers
function three_player_coop_adv2(;
  n_environment_sides=5,
  environment_radius=4,
  coupling_constraints=nothing,
  control_penalty=0.1,
  player_bounds=[[2, 5], [2, 5], [5, 10]],
  distance_metric=norm,
)
  cost = let
    function stage_cost(x, u, t, context_state)
      x1, x2, x3 = blocks(x)
      u1, u2, u3 = blocks(u)

      # distances
      v12 = x2[1:2] - x1[1:2]
      v13 = x3[1:2] - x1[1:2]
      v23 = x3[1:2] - x2[1:2]
      d12 = sqrt(distance_metric(v12) + 0.01)
      d13 = sqrt(distance_metric(v13) + 0.01)
      d23 = sqrt(distance_metric(v23) + 0.01)

      # directly from the lifted trajectory paper
      c1 = d13 + control_penalty * (distance_metric(u1) - distance_metric(u3))

      c2 = d23 + control_penalty * (distance_metric(u2) - distance_metric(u3))

      # evader should only depend on distances from pursuer
      c3 = -1 * (c1 + c2)

      # to promote distance between them for diverse area coverage
      # more distance between them => more area covered 
      # thinking of using 1/d12, so that higher distances are not supported
      c1 += 1/(d12 + 0.2)
      c2 += 1/(d12 + 0.2)

      [c1, c2, c3]
    end

    function reducer(scs)
      reduce(.+, scs) ./ length(scs)
    end

    TimeSeparableTrajectoryGameCost(stage_cost, reducer, GeneralSumCostStructure(), 1.0)
  end
  dynamics = ProductDynamics([get_dynamics_from_bound(bound[1], bound[2]) for bound in player_bounds] |> Tuple)
  env = PolygonEnvironment(n_environment_sides, environment_radius)
  TrajectoryGame(dynamics, cost, env, coupling_constraints)
end
