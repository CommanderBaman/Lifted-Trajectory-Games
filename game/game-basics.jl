# polygon environment 
using LazySets: LazySets
struct PolygonEnvironment{T}
  set::T
end

function PolygonEnvironment(sides::Int=4, radius=4)
  r = radius
  N = sides
  vertices = map(1:N) do n
    θ = 2π * n / N + pi / sides
    [r * cos(θ), r * sin(θ)]
  end
  PolygonEnvironment(vertices)
end

function PolygonEnvironment(vertices::AbstractVector{<:AbstractVector{<:Real}})
  PolygonEnvironment(LazySets.VPolytope(vertices))
end



# product dynamics
using BlockArrays: AbstractBlockArray
abstract type AbstractDynamics end
Base.@kwdef struct ProductDynamics{T} <: AbstractDynamics
  subsystems::T

  function ProductDynamics(subsystems::T) where {T}
    h = horizon(first(subsystems))
    all(sub -> horizon(sub) == h, subsystems) ||
      error("ProductDynamics can only be constructed from subsystems with the same horizon.")
    new{T}(subsystems)
  end
end

function (dynamics::ProductDynamics)(x::AbstractBlockArray, u::AbstractBlockArray, t=nothing)
  s = dynamics.subsystems[1]
  println(typeof(s))
  mortar([sub(x̂, u, t) for (sub, x̂, u) in zip(dynamics.subsystems, blocks(x), blocks(u))])
end


# zero sum cost structure
abstract type AbstractCostStructure end
struct ZeroSumCostStructure <: AbstractCostStructure end


# time separable trajectory game cost
struct TimeSeparableTrajectoryGameCost{T1,T2,T3}
  """
  A function `(x, u, t, context_state) -> sc` which maps the joint state `x` and input `u` for a
  given time step and surrounding context information `t` to a tuple of scalar costs `sc` for each
  player *at that time*.
  """
  stage_cost::T1
  """
  A function `(scs -> cs)` that reduces a sequence of stage cost tuples to a tuple of costs `cs`
  for all players. In the simplest case, this reduction operation may simply be the sum of elemnts
  (e.g. `reducer = scs -> reduce(.+, scs)`).
  """
  reducer::T2
  """
  An aditional structure hint for further optimization. See the docstring of
  `AbstractCostStructure` for further details.
  """
  structure::T3
  "A discount factor γ ∈ (0, 1] that exponentially decays the weight of future stage costs."
  discount_factor::Float64
end

function (c::TimeSeparableTrajectoryGameCost)(xs, us, context_state)
  ts = Iterators.eachindex(xs)
  Iterators.map(xs, us, ts) do x, u, t
    c.discount_factor^(t - 1) .* c.stage_cost(x, u, t, context_state)
  end |> c.reducer
end

# dynamics
using InfiniteArrays: ∞, Fill
using LinearAlgebra: norm

abstract type AbstractTemporalStructureTrait end
struct TimeVarying <: AbstractTemporalStructureTrait end
struct TimeInvariant <: AbstractTemporalStructureTrait end

Base.@kwdef struct LinearDynamics{TA,TB,TSB,TCB} <: AbstractDynamics
  A::TA
  B::TB
  "layout: (; lb::Vector{<:Real}, ub::Vector{<:Real})"
  state_bounds::TSB = (; lb=fill(-Inf, size(first(B), 1)), ub=fill(Inf, size(first(B), 1)))
  "layout (; lb::Vector{<:Real}, ub::Vector{<:Real})"
  control_bounds::TCB = (; lb=fill(-Inf, size(first(B), 2)), ub=fill(Inf, size(first(B), 2)))
end

function _block_diagonalize(::TimeInvariant, linear_subsystems, horizon)
  A = Fill(blockdiag([sparse(sub.A.value) for sub in linear_subsystems]...), horizon)

  B = let
    B_joint = blockdiag([sparse(sub.B.value) for sub in linear_subsystems]...)
    total_xdim = sum(state_dim, linear_subsystems)
    udims = map(control_dim, linear_subsystems)
    B_joint_blocked = BlockArray(B_joint, [total_xdim], udims)
    Fill(B_joint_blocked, horizon)
  end
  (; A, B)
end

function horizon(sys::LinearDynamics)
  length(sys.B)
end

function state_dim(dynamics::ProductDynamics)
  sum(state_dim(sub) for sub in dynamics.subsystems)
end

function control_dim(dynamcs::ProductDynamics)
  sum(control_dim(sub) for sub in dynamcs.subsystems)
end

function state_dim(sys::LinearDynamics)
  size(first(sys.A), 1)
end

function control_dim(sys::LinearDynamics)
  size(first(sys.B), 2)
end


function LinearDynamics(dynamics::ProductDynamics{<:AbstractVector{<:LinearDynamics}})
  (; A, B) = _block_diagonalize(
    temporal_structure_trait(dynamics),
    dynamics.subsystems,
    horizon(dynamics),
  )
  sb = state_bounds(dynamics)
  cb = control_bounds(dynamics)

  LinearDynamics(; A, B, state_bounds=sb, control_bounds=cb)
end

function time_invariant_linear_dynamics(; A, B, horizon=∞, bounds...)
  LinearDynamics(; A=Fill(A, horizon), B=Fill(B, horizon), bounds...)
end

function planar_double_integrator(; dt=0.1, m=1, kwargs...)
  dt2 = 0.5 * dt * dt
  # Layout is x := (x, y, px, py) and u := (Fx, Fy).

  time_invariant_linear_dynamics(;
    A=[
      1.0 0.0 dt 0.0
      0.0 1.0 0.0 dt
      0.0 0.0 1.0 0.0
      0.0 0.0 0.0 1.0
    ],
    B=[
      dt2 0.0
      0.0 dt2
      dt 0.0
      0.0 dt
    ] / m,
    kwargs...,
  )
end


# Trajectory game
Base.@kwdef struct TrajectoryGame{TD<:AbstractDynamics,TC,TE,TS}
  "An object that describes the dynamics of this trajectory game"
  dynamics::TD
  "A cost function taking (xs, us, [context]) with states `xs` and inputs `us` in Blocks and an
  optional `context` information. Returns a collection of cost values; one per player."
  cost::TC
  "The environment object that characerizes static constraints of the problem and can be used for
  visualization."
  env::TE
  "An object which encodes the constraints between different players. It must be callable as
  `con(xs, us) -> gs`: returning a collection of scalar constraints `gs` each of which is negative
  if the corresponding contraint is active."
  coupling_constraints::TS = nothing
end

# get number of players
function num_players(g::TrajectoryGame)
  num_players(g.dynamics)
end


function num_players(dynamics::ProductDynamics)
  length(dynamics.subsystems)
end


# get dynamics from bound 
function get_dynamics_from_bound(state_bound, control_bound)
  planar_double_integrator(;
    state_bounds=(; lb=[-Inf, -Inf, -state_bound, -state_bound], ub=[Inf, Inf, state_bound, state_bound]),
    control_bounds=(; lb=[-control_bound, -control_bound], ub=[control_bound, control_bound]),
  )
end