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


# main function
function rollout(
  dynamics,
  strategy,
  x1,
  T = horizon(dynamics);
  get_info = (γ, x, t) -> nothing,
  skip_last_strategy_call = false,
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
    up = strategy(xp, tt + 1)
    push!(us, up)

    # gather info
    infop = get_info(strategy, xs[tt], tt + 1)
    push!(infos, infop)
  end

  (; xs, us, infos)
end
