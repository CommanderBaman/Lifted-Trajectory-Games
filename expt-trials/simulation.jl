# dynamics call
function (dynamics::AbstractDynamics)(state, action::PrecomputedAction, t = nothing)
  if action.reference_state != state
      throw(
          ArgumentError("""
                        This precomputed action is only valid for states \
                        $(action.reference_state) but has been called for $state instead which \
                        will likely not produce meaningful results.
                        """),
      )
  end
  action.next_substate
end

# strategy calls

# strategy calls
function (strategy::RecedingHorizonStrategy)(state, time; evader_u_randomness = 3)
  plan_exists = !isnothing(strategy.receding_horizon_strategy)
  time_along_plan = time - strategy.time_last_updated + 1
  plan_is_still_valid = 1 <= time_along_plan <= strategy.turn_length

  update_plan = !plan_exists || !plan_is_still_valid
  # print("time:$time ")
  # update plan means learn from the current situation by calling solve_trajectory_game!
  if update_plan
    # print("update plan")
    initial_guess =
      strategy.generate_initial_guess(strategy.receding_horizon_strategy, state, time)
    warm_start_kwargs = isnothing(initial_guess) ? (;) : (; initial_guess)
    strategy.receding_horizon_strategy = solve_trajectory_game!(
      strategy.solver,
      strategy.game,
      state;
      strategy.solve_kwargs...,
      warm_start_kwargs...,
      evader_u_randomness = evader_u_randomness
    )
    strategy.time_last_updated = time
    time_along_plan = 1
  end
  # println()

  # this is not a recursive function but the call to the object returned by solve_trajectory_game! function
  strategy.receding_horizon_strategy(state, time_along_plan)
end


# main function
function rollout(
  dynamics,
  strategy,
  x1,
  T = horizon(dynamics);
  get_info = (γ, x, t) -> nothing,
  skip_last_strategy_call = false,
  evader_u_randomness = 3,
)
  xs = sizehint!([x1], T)
  us = sizehint!([strategy(x1, 1)], T)
  infos = sizehint!([get_info(strategy, x1, 1)], T)

  time_steps = 1:(T - 1)

  

  for tt in time_steps
    # get the new position
    xp = dynamics(xs[tt], us[tt], tt)
    push!(xs, xp)

    # skip last call
    if skip_last_strategy_call && tt == lastindex(time_steps)
        break
    end

    # get the next input
    up = strategy(xp, tt + 1; evader_u_randomness = evader_u_randomness)
    push!(us, up)

    # gather info
    infop = get_info(strategy, xs[tt], tt + 1)
    push!(infos, infop)
  end

  (; xs, us, infos)
end
