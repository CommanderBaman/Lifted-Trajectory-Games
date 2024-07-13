# get the basics
include("game-basics.jl")

# main function
function herd3_basic(;
  n_environment_sides=5,
  environment_radius=4,
  coupling_constraints=nothing,
  control_penalty=0.1,
  # array of [state_bounds, control_bounds] for each player
  player_bounds=[[5, 10], [1, 2], [1, 2]],
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
      xp, xe1, xe2 = blocks(x)
      # # not considering input for costs
      # up, ue... = blocks(u)

      # removing velocities 
      xp = xp[1:2]
      xe1 = xe1[1:2]
      xe2 = xe2[1:2]
      
      # mean of all evaders = their collection point
      evm = (xe1 + xe2) / 2

      # # xe - evm
      xe1_evm = xe1 - evm
      xe2_evm = xe2 - evm

      # # xe - xp
      xe1_xp = xe1 - xp
      xe2_xp = xe2 - xp
      
      # cost of pursuer is how closely they are herded 
      # and distance of herder from herd
      cp = herd_center_weight * distance_function(xp - evm)
      # sum of distance from evm of all evaders
      # cp += sum(map(x -> distance_function(x), xe_evm)) / length(xe)
      cp += herd_distribution_weight *  distance_function(xe1_evm) / 2
      cp += herd_distribution_weight *  distance_function(xe2_evm) / 2


      # # cost of evaders should be distance from pursuer
      # ce = map(x -> distance_function(x), xe_xp)
      ce1 = distance_function(xe1_xp)
      ce2 = distance_function(xe2_xp)

      [cp, -ce1, -ce2]
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
function herd3_adv1(;
  n_environment_sides=5,
  environment_radius=4,
  coupling_constraints=nothing,
  control_penalty=0.1,
  # array of [state_bounds, control_bounds] for each player
  player_bounds=[[5, 10], [1, 2], [1, 2]],
  distance_metric=norm,
  herd_center_weight = 0.1, 
  herd_distribution_weight = 1.8,
  herd_bonus=2
)
  cost = let
    function distance_function(v)
      sqrt(distance_metric(v) + 0.0000001)
    end
    function stage_cost(x, u, t, context_state)
      # first is pursuer, rest are evaders
      xp, xe1, xe2 = blocks(x)
      # # not considering input for costs
      # up, ue... = blocks(u)

      # removing velocities 
      xp = xp[1:2]
      xe1 = xe1[1:2]
      xe2 = xe2[1:2]
      
      # mean of all evaders = their collection point
      evm = (xe1 + xe2) / 2

      # # xe - evm
      xe1_evm = xe1 - evm
      xe2_evm = xe2 - evm

      # # xe - xp
      xe1_xp = xe1 - xp
      xe2_xp = xe2 - xp
      
      # cost of pursuer is how closely they are herded 
      # and distance of herder from herd
      cp = herd_center_weight * distance_function(xp - evm)
      # sum of distance from evm of all evaders
      # cp += sum(map(x -> distance_function(x), xe_evm)) / length(xe)
      cp += herd_distribution_weight * distance_function(xe1_evm) / 2
      cp += herd_distribution_weight * distance_function(xe2_evm) / 2


      # # cost of evaders should be distance from pursuer
      # ce = map(x -> distance_function(x), xe_xp)
      ce1 = distance_function(xe1_xp) 
      ce2 = distance_function(xe2_xp)
      
      # adding a bonus for collecting as a herd
      ce1 -= herd_bonus * distance_function(xe1_evm)
      ce2 -= herd_bonus * distance_function(xe2_evm)

      [cp, -ce1, -ce2]
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
