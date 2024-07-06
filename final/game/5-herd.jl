# get the basics
include("game-basics.jl")

# main function
function herd5_basic(;
  n_environment_sides=5,
  environment_radius=4,
  coupling_constraints=nothing,
  control_penalty=0.1,
  # array of [state_bounds, control_bounds] for each player
  player_bounds=[[5, 10], [1, 2], [1, 2], [1, 2], [1, 2]],
  distance_metric=norm,
)
  cost = let
    function distance_function(v)
      sqrt(distance_metric(v) + 0.0000001)
    end
    function stage_cost(x, u, t, context_state)
      # first is pursuer, rest are evaders
      xp, xe1, xe2, xe3, xe4 = blocks(x)
      # # not considering input for costs
      # up, ue... = blocks(u)

      # removing velocities 
      xp = xp[1:2]
      xe1 = xe1[1:2]
      xe2 = xe2[1:2]
      xe3 = xe3[1:2]
      xe4 = xe4[1:2]
      

      # mean of all evaders = their collection point
      evm = (xe1 + xe2 + xe3 + xe4) / 4

      # # xe - evm
      # xe_evm = xe .- Ref(evm)
      xe1_evm = xe1 - evm
      xe2_evm = xe2 - evm
      xe3_evm = xe3 - evm
      xe4_evm = xe4 - evm

      # # xe - xp
      # xe_xp = xe .- Ref(xp)
      xe1_xp = xe1 - xp
      xe2_xp = xe2 - xp
      xe3_xp = xe3 - xp
      xe4_xp = xe4 - xp
      
      # cost of pursuer is how closely they are herded 
      # and distance of herder from herd
      cp = distance_function(xp - evm)
      # sum of distance from evm of all evaders
      # cp += sum(map(x -> distance_function(x), xe_evm)) / length(xe)
      cp += distance_function(xe1_evm) /4
      cp += distance_function(xe2_evm) /4
      cp += distance_function(xe3_evm) /4
      cp += distance_function(xe4_evm) /4


      # # cost of evaders should be distance from pursuer
      # ce = map(x -> distance_function(x), xe_xp)
      ce1 = distance_function(xe1_xp)
      ce2 = distance_function(xe2_xp)
      ce3 = distance_function(xe3_xp)
      ce4 = distance_function(xe4_xp)

      # println("x: $x")
      # println("xp: $xp")
      # println("xe: $xe")
      # println("evm: $evm")
      # println("xe_evm: $xe_evm")
      # println("xe_xp: $xe_xp")
      # println("cp: $cp")
      # println("ce: $ce")
      # println("l = $(length(xe))")
      # throw(error("heheheheh"))
      [cp, -ce1, -ce2, -ce3, -ce4]
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
