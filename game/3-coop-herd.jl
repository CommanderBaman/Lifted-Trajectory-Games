# get the basics
include("game-basics.jl")

# main function
function herd_coop_3_basic(;
  n_environment_sides=5,
  environment_radius=4,
  coupling_constraints=nothing,
  control_penalty=0.1,
  # array of [state_bounds, control_bounds] for each player
  player_bounds=[[5, 10], [5, 10], [1, 2], [1, 2], [1, 2]],
  distance_metric=norm,
  herd_center_weight = 0.2, 
  herd_distribution_weight = 0.8,
)
  cost = let
    function distance_function(v)
      sqrt(distance_metric(v) + 0.0000001)
    end
    function stage_cost(x, u, t, context_state)
      # first is pursuer, rest are evaders
      xp1, xp2, xe1, xe2, xe3 = blocks(x)
      # # not considering input for costs
      # up, ue... = blocks(u)

      # removing velocities 
      xp1 = xp1[1:2]
      xp2 = xp2[1:2]
      xe1 = xe1[1:2]
      xe2 = xe2[1:2]
      xe3 = xe3[1:2]
      
      # mean of all evaders = their collection point
      evm = (xe1 + xe2 + xe3) / 3

      # # xe - evm
      # xe_evm = xe .- Ref(evm)
      xe1_evm = xe1 - evm
      xe2_evm = xe2 - evm
      xe3_evm = xe3 - evm

      # # xe - xp
      xe1_xp1 = xe1 - xp1
      xe2_xp1 = xe2 - xp1
      xe3_xp1 = xe3 - xp1

      xe1_xp2 = xe1 - xp2
      xe2_xp2 = xe2 - xp2
      xe3_xp2 = xe3 - xp2
      
      # cost of pursuer is how closely they are herded 
      # and distance of herder from herd
      cp1 = herd_center_weight * distance_function(xp1 - evm)
      # sum of distance from evm of all evaders
      cp1 += herd_distribution_weight * distance_function(xe1_evm) /3
      cp1 += herd_distribution_weight * distance_function(xe2_evm) /3
      cp1 += herd_distribution_weight * distance_function(xe3_evm) /3

      cp2 = herd_center_weight * distance_function(xp2 - evm)
      # sum of distance from evm of all evaders
      cp2 += herd_distribution_weight * distance_function(xe1_evm) /3
      cp2 += herd_distribution_weight * distance_function(xe2_evm) /3
      cp2 += herd_distribution_weight * distance_function(xe3_evm) /3

      # cooperative with other pursuers
      d12 = distance_function(xp1 - xp2)
      cp1 -= d12
      cp2 -= d12

      # # cost of evaders should be distance from pursuer
      # ce = map(x -> distance_function(x), xe_xp)
      ce1 = distance_function(xe1_xp1) + distance_function(xe1_xp2)
      ce2 = distance_function(xe2_xp1) + distance_function(xe2_xp2)
      ce3 = distance_function(xe3_xp1) + distance_function(xe3_xp2)

      [cp1, cp2, -ce1, -ce2, -ce3]
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




# main function
function herd_coop_3_adv1(;
  n_environment_sides=5,
  environment_radius=4,
  coupling_constraints=nothing,
  control_penalty=0.1,
  # array of [state_bounds, control_bounds] for each player
  player_bounds=[[5, 10], [5, 10], [1, 2], [1, 2], [1, 2]],
  distance_metric=norm,
  herd_center_weight = 0.2, 
  herd_distribution_weight = 0.8,
  herd_bonus=0.3
)
  cost = let
    function distance_function(v)
      sqrt(distance_metric(v) + 0.0000001)
    end
    function stage_cost(x, u, t, context_state)
      # first is pursuer, rest are evaders
      xp1, xp2, xe1, xe2, xe3 = blocks(x)
      # # not considering input for costs
      # up, ue... = blocks(u)

      # removing velocities 
      xp1 = xp1[1:2]
      xp2 = xp2[1:2]
      xe1 = xe1[1:2]
      xe2 = xe2[1:2]
      xe3 = xe3[1:2]
      
      # mean of all evaders = their collection point
      evm = (xe1 + xe2 + xe3) / 3

      # # xe - evm
      # xe_evm = xe .- Ref(evm)
      xe1_evm = xe1 - evm
      xe2_evm = xe2 - evm
      xe3_evm = xe3 - evm

      # # xe - xp
      xe1_xp1 = xe1 - xp1
      xe2_xp1 = xe2 - xp1
      xe3_xp1 = xe3 - xp1

      xe1_xp2 = xe1 - xp2
      xe2_xp2 = xe2 - xp2
      xe3_xp2 = xe3 - xp2
      
      # cost of pursuer is how closely they are herded 
      # and distance of herder from herd
      cp1 = herd_center_weight * distance_function(xp1 - evm)
      # sum of distance from evm of all evaders
      cp1 += herd_distribution_weight * distance_function(xe1_evm) /3
      cp1 += herd_distribution_weight * distance_function(xe2_evm) /3
      cp1 += herd_distribution_weight * distance_function(xe3_evm) /3

      cp2 = herd_center_weight * distance_function(xp2 - evm)
      # sum of distance from evm of all evaders
      cp2 += herd_distribution_weight * distance_function(xe1_evm) /3
      cp2 += herd_distribution_weight * distance_function(xe2_evm) /3
      cp2 += herd_distribution_weight * distance_function(xe3_evm) /3

      # cooperative with other pursuers
      d12 = distance_function(xp1 - xp2)
      cp1 -= d12
      cp2 -= d12

      # # cost of evaders should be distance from pursuer
      # ce = map(x -> distance_function(x), xe_xp)
      ce1 = distance_function(xe1_xp1) + distance_function(xe1_xp2)
      ce2 = distance_function(xe2_xp1) + distance_function(xe2_xp2)
      ce3 = distance_function(xe3_xp1) + distance_function(xe3_xp2)


      # adding a bonus for collecting as a herd
      ce1 -= herd_bonus * distance_function(xe1_evm)
      ce2 -= herd_bonus * distance_function(xe2_evm)
      ce3 -= herd_bonus * distance_function(xe3_evm)

      [cp1, cp2, -ce1, -ce2, -ce3]
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
