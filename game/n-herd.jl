# get the basics
include("game-basics.jl")

# main function
function n_herd_basic(;
  n_environment_sides=5,
  environment_radius=4,
  coupling_constraints=nothing,
  control_penalty=0.1,
  # array of [state_bounds, control_bounds] for each player
  player_bounds=[[5, 10], [5, 10]],
  distance_metric=norm,
)
  cost = let
    function distance_function(v)
      sqrt(distance_metric(v) + 0.0000001)
    end
    function stage_cost(x, u, t, context_state)
      # first is pursuer, rest are evaders
      xp, xe... = blocks(x)
      # # not considering input for costs
      # up, ue... = blocks(u)

      # removing velocities 
      xp = xp[1:2]
      xe = map(x -> x[1:2], xe)

      # mean of all evaders = their collection point
      evm = sum(xe) / length(xe)

      # xe - evm
      xe_evm = xe .- Ref(evm)

      # xe - xp
      xe_xp = xe .- Ref(xp)
      
      # cost of pursuer is how closely they are herded 
      # and distance of herder from herd
      cp = distance_function(xp - evm)
      # sum of distance from evm of all evaders
      # cp += sum(map(x -> distance_function(x), xe_evm)) / length(xe)
      # cost of evaders should be distance from pursuer
      ce = map(x -> distance_function(x), xe_xp)

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
      [cp, -ce...]
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
