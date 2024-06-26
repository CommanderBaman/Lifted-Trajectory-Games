# main struct 
abstract type AbstractPlayerStrategy end


struct JointStrategy{T1,T2} <: AbstractPlayerStrategy
  substrategies::T1
  info::T2
end

# a class to have multiple strategies in one place
function JointStrategy(substrategies)
  info = nothing
  JointStrategy(substrategies, info)
end


function (strategy::JointStrategy)(x, t=nothing)
  join_actions([sub(x, t) for sub in strategy.substrategies])
end

# for storing dynamics
struct PrecomputedAction{TS,TC,TN}
  reference_state::TS
  reference_control::TC
  next_substate::TN
end

function join_actions(actions::AbstractVector{<:PrecomputedAction})
  joint_reference_state = mortar([a.reference_state for a in actions])
  joint_reference_control = mortar([a.reference_control for a in actions])

  joint_next_state = mortar([a.next_substate for a in actions])
  PrecomputedAction(joint_reference_state, joint_reference_control, joint_next_state)
end
