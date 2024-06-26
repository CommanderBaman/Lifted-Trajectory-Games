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


#
struct GeneralSumCostStructure <: AbstractCostStructure end

# main function
function three_player_herd_meta_game(;
  n_environment_sides=5,
  environment_radius=4,
  coupling_constraints=nothing,
  control_penalty=0.1,
  dynamics=planar_double_integrator(;
    state_bounds=(; lb=[-Inf, -Inf, -5, -5], ub=[Inf, Inf, 5, 5]),
    control_bounds=(; lb=[-10, -10], ub=[10, 10]),
  ),
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

      # first is pursuer 
      # # this can be many things, 
      
      # # trying out a - b costheta
      # # Everyone vibes at their corner
      # a = max(d12, d13)
      # b = min(d12, d13)
      # cos_value = v12' * v13 / d12^2 / d13^2
      # c1 = a - b * cos_value
      
      # # trying out a + b 
      # # It doesn't herd, but the pursuer just chases one and gives up
      # c1 = d12 + d13
      
      # # trying out a - b, no reason, thinking about hyperbole or something
      # c1 = abs(d12 - d13)

      # # trying out distance between the two members, 
      # # Hoping the machine learning model learns the behavior
      # c1 = d23
      
      # # trying out the sum of all distances
      # c1 = d23 + d12 + d13

      # # trying out a + b costheta
      # # Everyone vibes at their corner
      # a = max(d12, d13)
      # b = min(d12, d13)
      # cos_value = v12' * v13 / d12^2 / d13^2
      # c1 = a + b * cos_value


      # # trying out the sum of all distances
      # # but penalizing it even more
      # c1 = (d23 + d12 + d13)^2
      
      
      # other two are evaders
      # directly from the lifted trajectory paper
      c2 =
        -1 * (d12 + control_penalty * (distance_metric(u1) - distance_metric(u2)))
      c3 =
        -1 * (d13 + control_penalty * (distance_metric(u1) - distance_metric(u3)))

      [c1, c2, c3]
    end

    function reducer(scs)
      reduce(.+, scs) ./ length(scs)
    end

    TimeSeparableTrajectoryGameCost(stage_cost, reducer, GeneralSumCostStructure(), 1.0)
  end
  # both the player have the same dynamics => replicating
  dynamics = ProductDynamics([dynamics for _ in 1:3] |> Tuple)
  env = PolygonEnvironment(n_environment_sides, environment_radius)
  TrajectoryGame(dynamics, cost, env, coupling_constraints)
end
