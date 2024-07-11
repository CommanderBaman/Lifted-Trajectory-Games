# qp solver
struct QPSolver end
is_thread_safe(::QPSolver) = true

"""
Solves quadratic program:
QP(y) := argmin_x 0.5 x'Qx + x'(Ry+q)
         s.t.  lb <= Ax + By <= ub
Additionally provides gradients ∇_y QP(y)

Q, R, A, and B should be sparse matrices of type SparseMatrixCSC.
q, a, and y should be of type Vector{Float64}.
"""
function solve(
  ::QPSolver,
  problem,
  x0,
  params::AbstractVector{<:AbstractFloat};
  initial_guess=nothing,
)
  (; cost_hess_rows, cost_hess_cols, parametric_cost_hess_vals) = problem.cost_hess
  (; jac_rows, jac_cols, parametric_jac_vals) = problem.jac_primals

  n = problem.n
  m = problem.num_equality + problem.num_inequality

  primals = zeros(n)
  duals = zeros(m)

  Qvals = zeros(size(cost_hess_rows, 1))
  parametric_cost_hess_vals(Qvals, params, primals)
  Q = sparse(cost_hess_rows, cost_hess_cols, Qvals, n, n)

  q = zeros(n)
  problem.parametric_cost_grad(q, params, primals)

  Avals = zeros(size(jac_rows, 1))
  parametric_jac_vals(Avals, x0, params, primals)
  A = sparse(jac_rows, jac_cols, Avals, m, n)

  cons = zeros(m)
  problem.parametric_cons(cons, x0, params, primals)

  lb = -cons
  ub = fill(Inf, length(lb))
  # why is upper bound bounded for equality reasons
  # this is bounded to ensure that 
  ub[1:(problem.num_equality)] .= lb[1:(problem.num_equality)]

  m = OSQP.Model()
  # changing verbose to true
  OSQP.setup!(m; P=sparse(Q), q=q, A=A, l=lb, u=ub, verbose=false, polish=true)

  if !isnothing(initial_guess)
    OSQP.warm_start!(m; x=initial_guess.x, y=initial_guess.y)
  end

  results = OSQP.solve!(m)
  if (results.info.status_val != 1)
    @warn "QP not cleanly solved. OSQP status is $(results.info.status_val)"
  end

  # println(axes(results.x))

  (;
    primals=results.x,
    equality_duals=-results.y[1:(problem.num_equality)],
    inequality_duals=-results.y[(problem.num_equality+1):end],
    info=(; raw_solution=results),
  )
end



# input reference parameter
Base.@kwdef struct InputReferenceParameterization
  α::Float64
end

function (parameterization::InputReferenceParameterization)(xs, us, params)
  horizon = length(us)
  # so basically we are ensuring that the parameters match the horizon of the 
  ps = reshape(params, :, horizon) |> eachcol
  sum(zip(xs, us, ps)) do (x, u, param)
    sum(0.5 .* parameterization.α .* u .^ 2 .- u .* param)
  end
end


# nn action generator 
using Flux: Flux, Chain, Dense, Optimise, @functor

struct NNActionGenerator{M,O,G}
  model::M
  optimizer::O
  n_actions::Int
  gradient_clipping_threshold::G
end
@functor NNActionGenerator (model,)

function NNActionGenerator(;
  input_dimension,
  parameter_dimension,
  n_actions,
  learning_rate,
  rng,
  initial_parameters,
  params_abs_max=10.0,
  hidden_dim=100,
  n_hidden_layers=2,
  output_activation=tanh,
  gradient_clipping_threshold=nothing,
)
  # generate neural networks
  if initial_parameters === :random
    init = (in, out) -> Flux.glorot_uniform(rng, in, out)
  elseif initial_parameters === :all_zero
    init = (in, out) -> zeros(in, out)
  else
    @assert false
  end
  model = Chain(
    Dense(input_dimension, hidden_dim, tanh; init),
    (Dense(hidden_dim, hidden_dim, tanh; init) for _ in 1:(n_hidden_layers-1))...,
    Dense(hidden_dim, parameter_dimension * n_actions, output_activation; init),
    x -> params_abs_max .* x,
  )
  optimizer = Optimise.Descent(learning_rate)
  NNActionGenerator(model, optimizer, n_actions, gradient_clipping_threshold)
end

function (g::NNActionGenerator)(x)
  stacked_goals = g.model(x)
  reshape(stacked_goals, :, g.n_actions)
end

# constraint handler  
struct LagrangianCouplingConstraintHandler
  violation_penalty::Float64
end

function (constraint_handler::LagrangianCouplingConstraintHandler)(game, xs, us, context_state)
  constraint_penalties = map(game.coupling_constraints) do coupling_constraints_per_player
    sum(coupling_constraints_per_player(xs, us)) do g
      if g >= 0
        # the constraint is already satsified, no penalty
        # return zero in same type as g
        zero(g)
      else
        -g * constraint_handler.violation_penalty
      end
    end
  end

  # lagrangian approximation to enforce coupling constraints
  constraint_penalties
end

# get constraints from the polygon environment
# so this one is different from the beneath 
function get_constraints(environment::PolygonEnvironment, player_index=nothing)
  # https://juliareach.github.io/LazySets.jl/dev/lib/API/#LazySets.API.constraints_list-Tuple{LazySets.API.LazySet}
  constraints = LazySets.constraints_list(environment.set)
  function (state)
    positions = (substate[1:2] for substate in blocks(state))
    mapreduce(vcat, Iterators.product(constraints, positions)) do (constraint, position)
      # a linear constraint saying the point must be on the same side as origin
      -constraint.a' * position + constraint.b
    end
  end
end

# returns a function in which the input gets bound between the values provided
# empty array if constraints not satisfied 
# array - bound if constraints satisfied
function get_constraints_from_box_bounds(bounds)
  function (y)
    mapreduce(vcat, [(bounds.lb, 1), (bounds.ub, -1)]) do (bound, sign)
      # drop constraints for unbounded variables
      mask = (!isinf).(bound)
      sign * (y[mask] - bound[mask])
    end
  end
end


# execution policy 
abstract type AbstractExecutionPolicy end
struct SequentialExecutionPolicy <: AbstractExecutionPolicy end
struct MultiThreadedExecutionPolicy <: AbstractExecutionPolicy end

# bounds 

function state_bounds(sys::LinearDynamics)
  sys.state_bounds
end
function control_bounds(sys::LinearDynamics)
  sys.control_bounds
end

function state_bounds(dynamics::ProductDynamics)
  _mortar_bounds(dynamics, state_bounds)
end

function control_bounds(dynamics::ProductDynamics)
  _mortar_bounds(dynamics, control_bounds)
end

function _parameter_dimension(::InputReferenceParameterization; horizon, state_dim, control_dim)
  horizon * control_dim
end


# parametric trajectory optimization problem
using Symbolics: Symbolics, @variables, scalarize
using SparseArrays: findnz
using BlockArrays: blocks


Base.@kwdef struct ParametricTrajectoryOptimizationProblem{T1,T2,T3,T4,T5,T6,T7,T8,T9}
  # https://github.com/JuliaLang/julia/issues/31231
  horizon::Int
  n::Int
  state_dim::Int
  control_dim::Int
  parameter_dim::Int
  num_equality::Int
  num_inequality::Int
  parametric_cost::T1
  parametric_cost_grad::T2
  parametric_cost_jac::T3
  parametric_cons::T4
  jac_primals::T5
  jac_params::T6
  cost_hess::T7
  lag_hess_primals::T8
  lag_jac_params::T9
end

function ParametricTrajectoryOptimizationProblem(
  cost,
  dynamics,
  inequality_constraints,
  state_dim,
  control_dim,
  parameter_dim,
  horizon;
  parameterize_dynamics=false,
)
  # there is no math behind this. Just variables that define the problem
  # total length of array
  n = horizon * (state_dim + control_dim)
  num_equality = nx = horizon * state_dim

  # Point: what is the difference between equality and inequality constraints
  # equality constraints force x(t+1) = Ax + Bu 
  # inequality constraints force the environment, state and control limits

  # define symbolic variables
  # scalarize turns all the array elements into their own symbols 
  x0, z, p = let
    @variables(x0[1:state_dim], z[1:n], p[1:parameter_dim]) .|> scalarize
  end

  #   get the state variables array 
  xs = hcat(x0, reshape(z[1:nx], state_dim, horizon)) |> eachcol |> collect
  us = reshape(z[(nx+1):n], control_dim, horizon) |> eachcol |> collect

  # cost and its jacobian
  # for cost see InputReferenceParameterization
  cost_val = cost(xs[2:end], us, p)
  cost_grad = Symbolics.gradient(cost_val, z)
  cost_jac_param = Symbolics.sparsejacobian(cost_grad, p)
  # hover over findnz to find what it does
  # basically a sparse matrix implementation
  (cost_jac_rows, cost_jac_cols, cost_jac_vals) = findnz(cost_jac_param)

  constraints_val = Symbolics.Num[]
  # NOTE: The dynamics constraints **must** always be first since the backward pass exploits this
  # structure to more easily identify active constraints.
  dynamics_parameterized = parameterize_dynamics ? dynamics : (x, u, t, _) -> dynamics(x, u, t)
  for t in eachindex(us)
    append!(constraints_val, dynamics_parameterized(xs[t], us[t], t, p) .- xs[t+1])
  end
  append!(constraints_val, inequality_constraints(xs[2:end], us, p))

  num_inequality = length(constraints_val) - num_equality

  con_jac = Symbolics.sparsejacobian(constraints_val, z)
  (jac_rows, jac_cols, jac_vals) = findnz(con_jac)

  con_jac_p = Symbolics.sparsejacobian(constraints_val, p)
  (jac_p_rows, jac_p_cols, jac_p_vals) = findnz(con_jac_p)

  num_constraints = length(constraints_val)

  # lagrangian?
  λ, cost_scaling, constraint_penalty_scaling = let
    @variables(λ[1:num_constraints], cost_scaling, constraint_penalty_scaling) .|> scalarize
  end
  lag = cost_scaling * cost_val - constraint_penalty_scaling * λ' * constraints_val
  lag_grad = Symbolics.gradient(lag, z)

  lag_hess = Symbolics.sparsejacobian(lag_grad, z)
  lag_jac = Symbolics.sparsejacobian(lag_grad, p)
  expression = Val{false}
  (lag_hess_rows, lag_hess_cols, hess_vals) = findnz(lag_hess)
  (lag_jac_rows, lag_jac_cols, lag_jac_vals) = findnz(lag_jac)

  parametric_cost = let
    # https://docs.sciml.ai/Symbolics/stable/manual/build_function/#Symbolics.build_function
    cost_fn = Symbolics.build_function(cost_val, [p; z]; expression)
    (params, primals) -> cost_fn(vcat(params, primals))
  end

  parametric_cost_grad = let
    cost_grad_fn! = Symbolics.build_function(cost_grad, [p; z]; expression)[2]
    (grad, params, primals) -> cost_grad_fn!(grad, vcat(params, primals))
  end

  cost_hess = let
    cost_hess_sym = Symbolics.sparsejacobian(cost_grad, z)
    (cost_hess_rows, cost_hess_cols, cost_hess_vals) = findnz(cost_hess_sym)
    cost_hess_fn! = Symbolics.build_function(cost_hess_vals, [p; z]; expression)[2]
    parametric_cost_hess_vals =
      (hess, params, primals) -> cost_hess_fn!(hess, vcat(params, primals))
    (; cost_hess_rows, cost_hess_cols, parametric_cost_hess_vals)
  end

  parametric_cost_jac_vals = let
    cost_jac_param_fn! = Symbolics.build_function(cost_jac_vals, [p; z]; expression)[2]
    (vals, params, primals) -> cost_jac_param_fn!(vals, vcat(params, primals))
  end

  parametric_cons = let
    con_fn! = Symbolics.build_function(constraints_val, [x0; p; z]; expression)[2]
    (cons, x0, params, primals) -> con_fn!(cons, vcat(x0, params, primals))
  end

  parametric_jac_vals = let
    jac_vals_fn! = Symbolics.build_function(jac_vals, [x0; p; z]; expression)[2]
    (vals, x0, params, primals) -> jac_vals_fn!(vals, vcat(x0, params, primals))
  end

  parametric_jac_p_vals = let
    jac_p_vals_fn! = Symbolics.build_function(jac_p_vals, [x0; p; z]; expression)[2]
    (vals, x0, params, primals) -> jac_p_vals_fn!(vals, vcat(x0, params, primals))
  end

  parametric_lag_hess_vals = let
    hess_vals_fn! = Symbolics.build_function(
      hess_vals,
      [x0; p; z; λ; cost_scaling; constraint_penalty_scaling];
      expression,
    )[2]
    (vals, x0, params, primals, duals, cost_scaling, constraint_penalty_scaling) ->
      hess_vals_fn!(
        vals,
        vcat(x0, params, primals, duals, cost_scaling, constraint_penalty_scaling),
      )
  end

  parametric_lag_jac_vals = let
    ∇lac_jac_vals_fn! = Symbolics.build_function(
      lag_jac_vals,
      vcat(x0, p, z, λ, cost_scaling, constraint_penalty_scaling);
      expression,
    )[2]
    (vals, x0, params, primals, duals, cost_scaling, constraint_penalty_scaling) ->
      ∇lac_jac_vals_fn!(
        vals,
        vcat(x0, params, primals, duals, cost_scaling, constraint_penalty_scaling),
      )
  end

  parametric_cost_jac = (; cost_jac_rows, cost_jac_cols, parametric_cost_jac_vals)
  jac_primals = (; jac_rows, jac_cols, parametric_jac_vals)
  jac_params = (; jac_p_rows, jac_p_cols, parametric_jac_p_vals)
  lag_hess_primals = (; lag_hess_rows, lag_hess_cols, parametric_lag_hess_vals)
  lag_jac_params = (; lag_jac_rows, lag_jac_cols, parametric_lag_jac_vals)

  # So basically this part is generating code whose values will be filled later. 
  # kind of like allocating space for variables using the Symbolics library
  ParametricTrajectoryOptimizationProblem(;
    horizon,
    n,
    state_dim,
    control_dim,
    parameter_dim,
    num_equality,
    num_inequality,
    parametric_cost,
    parametric_cost_grad,
    parametric_cost_jac,
    parametric_cons, # parametric constraints
    jac_primals,  # jacobian of constraints (wrt z)
    jac_params, # jacobian of paramers (wrt p)
    cost_hess, # hessian of cost
    lag_hess_primals,
    lag_jac_params,
  )
end

function parameter_dimension(problem::ParametricTrajectoryOptimizationProblem)
  problem.parameter_dim
end

# dynamics calls 

function (sys::LinearDynamics)(x, u, t::Int)
  sys.A[t] * x + sys.B[t] * u
end
function (sys::LinearDynamics)(x, u, ::Nothing=nothing)
  temporal_structure_trait(sys) isa TimeInvariant ||
    error("Only time-invariant systems can ommit the `t` argument.")
  sys.A.value * x + sys.B.value * u
end

# optimizer 

struct Optimizer{TP<:ParametricTrajectoryOptimizationProblem,TS}
  problem::TP
  solver::TS
end

parameter_dimension(optimizer::Optimizer) = parameter_dimension(optimizer.problem)



# main struct
mutable struct LiftedTrajectoryGameSolver{T1,T2,T3,T4,T5,T6,T7,T8,T9,T10}
  "A collection of action generators, one for each player in the game."
  trajectory_reference_generators::T1
  "A collection of trajectory optimizers, one for each player in the game"
  trajectory_optimizers::T2
  "A callable `(game, xs, us) -> cs` which maps the joint state-input trajectory `(xs, us)` to a
  tuple of scalar costs `cs` for a given `game`. In the simplest case, this may just forward to
  the `game.cost`. More generally, however, this function will add penalties to enforce
  constraints."
  coupling_constraints_handler::T3
  "The number of time steps to plan into the future."
  planning_horizon::T4
  "A random number generator to generate non-deterministic strategies."
  rng::T5
  "How much affect the dual regularization has on the costs"
  dual_regularization_weights::T6
  "A flag that can be set to enable/disable learning"
  enable_learning::T7
  "An AbstractExecutionPolicy that determines whether the solve is computed in parallel or
  sequentially."
  execution_policy::T8
  "A state value predictor (e.g. a neural network) that maps the current state to a tuple of
  optimal cost-to-go's for each player."
  state_value_predictor::T9
  "A function to compose the input of the reference generator from parameters (player_i, state, context)."
  compose_reference_generator_input::T10
end

"""
Convenience constructor to derive a suitable solver directly from a given game.
"""
# main function
using Random

# define lifted solver
function define_lifted_trajectory_solver(
  game::TrajectoryGame{<:ProductDynamics},
  planning_horizon;
  trajectory_parameterizations=[
    InputReferenceParameterization(; α=3) for _ in 1:num_players(game)
  ],
  trajectory_solver=QPSolver(),
  rng=Random.MersenneTwister(1),
  context_dimension=0,
  reference_generator_input_dimension=state_dim(game.dynamics) + context_dimension,
  # this :random creates a symbol in julia, a special type of data
  initial_parameters=[:random for _ in 1:num_players(game)],
  n_actions=[2 for _ in 1:num_players(game)],
  learning_rates=[0.05 for _ in 1:num_players(game)],
  reference_generator_constructors=[NNActionGenerator for _ in 1:num_players(game)],
  gradient_clipping_threshold=nothing,
  coupling_constraints_handler=LagrangianCouplingConstraintHandler(100),
  dual_regularization_weights=[1e-4 for _ in 1:num_players(game)],
  enable_learning=[true for _ in 1:num_players(game)],
  execution_policy=SequentialExecutionPolicy(),
  state_value_predictor=nothing,
  compose_reference_generator_input=(i, game_state, context) -> [game_state; context],
)
  # define trajectory optimizers for all players
  # spits out ([trajectory_problem, trajectory_solver]) for each player
  trajectory_optimizers = let
    map(
      game.dynamics.subsystems, # this contain dynamics for each player
      trajectory_parameterizations, # this is the cost function based on inputs
      Iterators.countfrom(),
    ) do subdynamics, parameterization, player_index

      # this returns a function which will always let you be within bounds
      inequality_constraints = let
        environment_constraints = get_constraints(game.env, player_index)
        state_box_constraints =
          get_constraints_from_box_bounds(state_bounds(subdynamics))
        control_box_constraints =
          get_constraints_from_box_bounds(control_bounds(subdynamics))

        function (xs, us, params)
          pc = mapreduce(x -> environment_constraints(x), vcat, xs)
          sc = mapreduce(state_box_constraints, vcat, xs)
          cc = mapreduce(control_box_constraints, vcat, us)
          [pc; sc; cc]
        end
      end


      trajectory_problem = ParametricTrajectoryOptimizationProblem(
        parameterization,
        subdynamics,
        inequality_constraints,
        state_dim(subdynamics),
        control_dim(subdynamics),
        _parameter_dimension(
          parameterization;
          horizon=planning_horizon,
          state_dim=state_dim(subdynamics),
          control_dim=control_dim(subdynamics),
        ),
        planning_horizon,
      )
      # just a packaging
      Optimizer(trajectory_problem, trajectory_solver)
    end
  end

  # safeguard for multithreaded execution
  if execution_policy isa MultiThreadedExecutionPolicy &&
     any(!is_thread_safe, trajectory_optimizers)
    throw(
      ArgumentError(
        """
        The solver trajectory optimization backend that you selected does not support \
        multi-threaded execution. Consider using a another backend or disable \
        multi-threading by handing another `execution_policy`.
        """,
      ),
    )
  end

  trajectory_reference_generators = map(
    reference_generator_constructors, # neural network constructor
    trajectory_optimizers, # you just went through this
    n_actions, # how many trajectories do we need to learn, IMPT: action = trajectory 
    initial_parameters, # initial parameters would be random or zero
    learning_rates, # learning rate of model
  ) do constructor, trajectory_optimizer, n_actions, initial_parameters, learning_rate
    constructor(;
      input_dimension=reference_generator_input_dimension,
      parameter_dimension=parameter_dimension(trajectory_optimizer),
      n_actions,
      learning_rate,
      rng,
      initial_parameters,
      gradient_clipping_threshold,
    )
  end
  # send all the results back in an object
  LiftedTrajectoryGameSolver(
    trajectory_reference_generators,
    trajectory_optimizers,
    coupling_constraints_handler,
    planning_horizon,
    rng,
    dual_regularization_weights,
    enable_learning,
    execution_policy,
    state_value_predictor,
    compose_reference_generator_input,
  )
end

# compute costs
function compute_costs(
  solver,
  trajectories_per_player, # (xs, us, λs=inequality_duals, info)
  context_state;
  trajectory_pairings,
  n_players,
  game,
)
  cost_tensor = map_threadable(trajectory_pairings, solver.execution_policy) do i
    # get the trajectories in (xs, us, λs=inequality_duals, info)
    trajectories = (trajectories_per_player[j][i[j]] for j in 1:n_players)

    cost_horizon =
    # state_value_predictor = nothing or NeuralStateValuePredictor
      isnothing(solver.state_value_predictor) ? solver.planning_horizon + 1 :
      solver.state_value_predictor.turn_length

    # extract information from trajectory
    xs = map((t.xs[1:cost_horizon] for t in trajectories)...) do x...
      mortar(collect(x))
    end

    us = map((t.us[1:(cost_horizon-1)] for t in trajectories)...) do u...
      mortar(collect(u))
    end

    # calculate costs 
    # sqrt(norm(x1-x2) + 0.1) + 0.1 (norm(u1) - norm(u2))
    trajectory_costs = game.cost(xs, us, context_state)
    if !isnothing(game.coupling_constraints)
      trajectory_costs .+= solver.coupling_constraints_handler(game, xs, us, context_state)
    end

    if !isnothing(solver.state_value_predictor)
      trajectory_costs .+=
        game.cost.discount_factor .* solver.state_value_predictor(xs[cost_horizon])
    end

    trajectory_costs
  end

  # transpose tensor of tuples to tuple of tensors
  map(1:n_players) do player_i
    map(cost_tensor) do pairing
      pairing[player_i]
    end
  end
end

function map_threadable(f, collection, ::SequentialExecutionPolicy)
  map(f, collection)
end

function map_threadable(f, collection, ::MultiThreadedExecutionPolicy)
  ThreadsX.map(f, collection)
end


# TRAJ = optimize trajectories
function optimize_trajectories(
  solver,
  initial_state,
  # references are ξ and there are n number of them stacked on each other
  stacked_references; # [ξ1, ξ2 ... ξ_num_players]
  n_players,
  trajectory_pairings,
)
  # break mortar on initial state
  state_per_player = blocks(initial_state)

  # basically separating ξ for each player into an array of ξ with nice iterators?
  # b = let... b[1] = ξ1 and so on
  references_per_player = let
    # get a iterable wrapper around both the ξ
    iterable_references = Iterators.Stateful(eachcol(stacked_references))
    map(1:n_players) do ii
      n_references = size(trajectory_pairings, ii)
      collect(Iterators.take(iterable_references, n_references))
    end
  end

  map_threadable(1:n_players, solver.execution_policy) do player_i
    references = references_per_player[player_i] # ξ
    trajectory_optimizer = solver.trajectory_optimizers[player_i] # OSQP?
    substate = state_per_player[player_i]
    map(reference -> trajectory_optimizer(
        substate, # x0 in function, x_i in real
        reference # ξ, parameters in function
      ), references)
  end
end


# optimizer calls
using BlockArrays: reshape
function (optimizer::Optimizer)(x0, params; initial_guess=nothing)
  @assert length(x0) == optimizer.problem.state_dim
  sol = solve(optimizer.solver, optimizer.problem, x0, params; initial_guess)
  (; horizon, state_dim, control_dim) = optimizer.problem
  # series of x = [x(0), x(1), ... x(horizon)]
  xs = [[x0]; collect.(eachcol(reshape(sol.primals[1:(horizon*state_dim)], state_dim, :)))]
  # series of u = [u(0), u(1), ... u(horizon)]
  us = collect.(eachcol(reshape(sol.primals[((horizon*state_dim)+1):end], control_dim, :)))
  (; xs, us, λs=sol.inequality_duals, sol.info)
end

# solve calls 
using ForwardDiff: ForwardDiff
using SparseArrays: sparse #, spzeros
using OSQP: OSQP

function solve(
  solver, # trajectory solver defined in optimizer = QPSolver 
  problem, # trajectory problem  = parametric trajectory optimization problem
  x0, # current state of object = x
  params::AbstractVector{<:ForwardDiff.Dual{T}}; #  ξ
  kwargs...,
) where {T}
  # strip off the duals:
  # remove any partials if present in the ξ
  params_v = ForwardDiff.value.(params)
  params_d = ForwardDiff.partials.(params)
  # forward pass
  # this solve is from the lifted trajectory game solver, around line 14
  res = solve(solver, problem, x0, params_v; kwargs...)
  # backward pass
  # I don't know why we are calculating these except for the ∂inequality_duals
  _back = _solve_pullback(solver, res, problem, x0, params_v)

  ∂primals = _back.∂x∂y * params_d
  ∂inequality_duals = _back.∂duals∂y * params_d

  # glue forward and backward pass together into dual number types
  (;
    primals=ForwardDiff.Dual{T}.(res.primals, ∂primals),
    # we don't need these so I'm just creating a non-dual result size here
    res.equality_duals,
    inequality_duals=ForwardDiff.Dual{T}.(res.inequality_duals, ∂inequality_duals),
    res.info,
  )
end


# pullback solves
using SparseArrays: spzeros
using LinearAlgebra: qr, I

function _solve_pullback(solver, res, problem, x0, params)
  # lagrangian hessian z^2
  (; lag_hess_rows, lag_hess_cols, parametric_lag_hess_vals) = problem.lag_hess_primals
  # lagrangian jacobian pz 
  (; lag_jac_rows, lag_jac_cols, parametric_lag_jac_vals) = problem.lag_jac_params

  (; jac_rows, jac_cols, parametric_jac_vals) = problem.jac_primals
  (; jac_p_rows, jac_p_cols, parametric_jac_p_vals) = problem.jac_params

  (; n, num_equality) = problem
  m = problem.num_equality + problem.num_inequality
  l = size(params, 1)

  (; primals, equality_duals, inequality_duals) = res
  duals = [equality_duals; inequality_duals]

  Qvals = zeros(size(lag_hess_rows, 1))
  parametric_lag_hess_vals(Qvals, x0, params, primals, duals, 1.0, 1.0)
  # hessian of lagrangian wrt z = [x; u]
  Q = sparse(lag_hess_rows, lag_hess_cols, Qvals, n, n)

  Rvals = zeros(size(lag_jac_rows, 1))
  parametric_lag_jac_vals(Rvals, x0, params, primals, duals, 1.0, 1.0)
  # jacobian of (gradient of lagrangian wrt z) wrt ξ
  R = sparse(lag_jac_rows, lag_jac_cols, Rvals, n, l)

  Avals = zeros(size(jac_rows, 1))
  parametric_jac_vals(Avals, x0, params, primals)
  # jacobian of constraints wrt z
  A = sparse(jac_rows, jac_cols, Avals, m, n)

  Bvals = zeros(size(jac_p_rows, 1))
  parametric_jac_p_vals(Bvals, x0, params, primals)
  # jacobian of constrains wrt p
  B = sparse(jac_p_rows, jac_p_cols, Bvals, m, l)

  # checking which equalities and inequalities (constraints) are active

  # first inequalities. consider those which are > 0.001 
  lower_active = duals .> 1e-3
  # ignore the equality ones here
  lower_active[1:num_equality] .= 0
  # equalities
  equality = zero(lower_active)
  equality[1:num_equality] .= 1
  # compose the two
  active = lower_active .| equality
  # get how many constraints are active
  num_lower_active = sum(lower_active)

  # get the corresponding rows of active constraints
  A_l_active = A[lower_active, :]
  A_equality = A[equality, :]
  B_l_active = B[lower_active, :]
  B_equality = B[equality, :]
  A_active = [A_equality; A_l_active]
  B_active = [B_equality; B_l_active]

  dual_inds = eachindex(duals)
  lower_active_map = dual_inds[lower_active] .- num_equality

  M = [
    Q -A_active'
    A_active 0I
  ]
  N = [R; B_active]

  # TODO: Understand how this calculation is going on
  # MinvN = inv(-M) * N
  MinvN = qr(-M) \ Matrix(N)
  ∂x∂y = MinvN[1:n, :]
  ∂duals∂y = spzeros(length(inequality_duals), length(params))
  ∂duals∂y[lower_active_map, :] .= let
    lower_dual_range = (1:num_lower_active) .+ (n + num_equality)
    MinvN[lower_dual_range, :]
  end

  (; ∂x∂y, ∂duals∂y)
end

# π = generate trajectory references
function generate_trajectory_references(solver, initial_state, context_state; n_players)
  map(1:n_players) do player_i
    # compose_reference_generator_input currently just gives you the game state, which is the initial state here
    # line 481 in lifted_trajectory_game_solver
    # (a,b,c) -> [b;c]
    input = solver.compose_reference_generator_input(player_i, initial_state, context_state)
    solver.trajectory_reference_generators[player_i](input)
  end
end

# cost structure 
cost_structure_trait(c::TimeSeparableTrajectoryGameCost) = c.structure

# cost gradients
function cost_gradients(back, solver, game)
  cost_gradients(back, solver, game, cost_structure_trait(game.cost))
end


struct GeneralSumCostStructure <: AbstractCostStructure end
function cost_gradients(back, solver, game, ::GeneralSumCostStructure)
  n_players = num_players(game)
  ∇L = map(1:n_players) do n
    if solver.enable_learning[n]
      loss_per_player = [i == n ? 1 : 0 for i in 1:n_players]
      back((; loss_per_player, info=nothing)) |> copy
    else
      nothing
    end
  end
end

function cost_gradients(back, solver, game, ::ZeroSumCostStructure)
  num_players(game) == 2 || error("Not implemented for N>2 players")
  if !isnothing(game.coupling_constraints)
    return cost_gradients(back, solver, game, GeneralSumCostStructure())
  end
  if !any(solver.enable_learning)
    return ∇L = (nothing, nothing)
  end
  # TODO: Why do we give this input to the gradient calculator
  # function forward_pass(; solver, game, initial_state, context_state, min_action_probability)
  # I don't think this means anything
  # Source: https://fluxml.ai/Zygote.jl/stable/adjoints
  ∇L_1 = back((; loss_per_player=[1, 0], info=nothing))

  ∇L = [∇L_1, -1 .* ∇L_1]
end


# compute regularized loss
function compute_regularized_loss(
  trajectories_per_player, # (xs, us, λs=inequality_duals, info)
  game_value_per_player;
  n_players,
  planning_horizon,
  dual_regularization_weights, # default = [1e-4 for _ in 1:num_players(game)],
)
  # huber loss function
  # https://en.wikipedia.org/wiki/Huber_loss
  function huber(x; δ=1)
    if abs(x) > δ
      δ * (abs(x) - 0.5δ) # this is allowed in julia
    else
      0.5x^2
    end
  end

  dual_regularizations =
    [sum(sum(huber.(c.λs)) for c in trajectories_per_player[i]) for i in 1:n_players] ./ planning_horizon

  # so regularization is done via the constraint duals 
  loss_per_player = [
    game_value_per_player[i] + dual_regularization_weights[i] * dual_regularizations[i] for
    i in 1:n_players
  ]
end


# clean tuple 
function clean_info_tuple(; game_value_per_player, mixing_strategies, trajectories_per_player)
  (;
    game_value_per_player=ForwardDiff.value.(game_value_per_player),
    mixing_strategies=[ForwardDiff.value.(q) for q in mixing_strategies],
    trajectories_per_player=map(trajectories_per_player) do trajectories
      map(trajectories) do trajectory
        (;
          xs=[ForwardDiff.value.(x) for x in trajectory.xs],
          us=[ForwardDiff.value.(u) for u in trajectory.us],
          λs=ForwardDiff.value.(trajectory.λs),
        )
      end
    end,
  )
end


# forward pass 
function forward_pass(; solver, game, initial_state, context_state, min_action_probability)

  # specifically how the things are calculated 
  # just do one loop and you will be done
  # calculate one loop by hand, don't forget about the dimension. note them
  n_players = num_players(game)
  # ξ = π(θ)
  references_per_player =
  # calls the neural networks defined in lifted trajectory game solver using the initial state
    generate_trajectory_references(solver, initial_state, context_state; n_players)

  # stack all the generated references 
  stacked_references = reduce(hcat, references_per_player)

  # define some local variables
  local trajectories_per_player, mixing_strategies, game_value_per_player


  loss_per_player = Zygote.forwarddiff(
    stacked_references;
    chunk_threshold=length(stacked_references),
  ) do stacked_references

    # iterators product forms a iterator to loop over the references 
    # collect forms a array according to type contained within it
    # currently (1, 1)
    trajectory_pairings =
      Iterators.product([axes(references, 2) for references in references_per_player]...) |> collect
    # τ = TRAJ(ξ)
    # we get (xs, us, λs=inequality_duals, info)
    trajectories_per_player = optimize_trajectories(
      solver,
      initial_state,
      stacked_references;
      n_players,
      trajectory_pairings,
    )

    # get the cost of each player: sum(f)
    # Evaluate the functions on all joint trajectories in the cost tensor
    # gets [c -c]
    cost_tensors_per_player = compute_costs(
      solver,
      trajectories_per_player,
      context_state;
      trajectory_pairings,
      n_players,
      game,
    )

    # Compute the mixing strategies, q_i, via a finite game solve;
    # https://github.com/forrestlaine/TensorGames.jl
    # This is the BMG
    # solves for nash equilibrium 
    # gives out q_i
    # TODO: understand math behind this equilibrium game
    mixing_strategies =
      TensorGames.compute_equilibrium(cost_tensors_per_player; ϵ=min_action_probability).x
    # L
    # this is equillibrium cost
    game_value_per_player = [
      TensorGames.expected_cost(mixing_strategies, cost_tensor) for
      cost_tensor in cost_tensors_per_player
    ]

    # this also considers the λs=inequality_duals in the loss function
    compute_regularized_loss(
      trajectories_per_player,
      game_value_per_player;
      n_players,
      solver.planning_horizon,
      solver.dual_regularization_weights,
    )
  end

  # strip of dual number types for downstream operation
  info = clean_info_tuple(; game_value_per_player, mixing_strategies, trajectories_per_player)

  (; loss_per_player, info)
end


# another neural network
struct NeuralStateValuePredictor{T1,T2,T3}
  model::T1
  optimizer::T2
  replay_buffer::T3
  turn_length::Int
  batch_size::Int
  n_epochs_per_update::Int # how many times to update the model everytime we get the game value (/run simulation)
end

@functor NeuralStateValuePredictor (model,)

# function NeuralStateValuePredictor(;
#     game,
#     learning_rate,
#     rng,
#     turn_length,
#     replay_buffer=NamedTuple[],
#     output_scaling=1,
#     n_hidden_layers=4,
#     hidden_dim=100,
#     activation=leakyrelu,
#     batch_size=50,
#     n_epochs_per_update=10,
# )
#     # defining a function for initialization of weights
#     init(in, out) = Flux.glorot_uniform(rng, in, out)

#     model = let
#         legs = map(1:num_players(game)) do ii
#             Chain(
#                 Dense(state_dim(game.dynamics), hidden_dim, activation; init),
#                 (
#                     Dense(hidden_dim, hidden_dim, activation; init) for
#                     _ in 1:(n_hidden_layers-1)
#                 )...,
#                 Dense(hidden_dim, 1; init),
#                 x -> output_scaling * x,
#                 only,
#             )
#         end
#         Split(legs)
#     end

#     optimizer = Optimise.Descent(learning_rate)

#     NeuralStateValuePredictor(
#         model,
#         optimizer,
#         replay_buffer,
#         turn_length,
#         batch_size,
#         n_epochs_per_update,
#     )
# end

function (p::NeuralStateValuePredictor)(state)
  joint_state = reduce(vcat, state)
  p.model(joint_state)
end


# process gradients so that they won't go out of bounds
function preprocess_gradients!(∇, reference_generator::NNActionGenerator, θ; kwargs...)
  # if there is a gradient clipping threshold 
  # scale it to the threshold
  if !isnothing(reference_generator.gradient_clipping_threshold)
    # find maximum absolute value (=v) in θ
    v = maximum(θ) do p
      maximum(g -> abs(g), ∇[p])
    end

    # divide all of the numbers in θ if v > threshold
    if v > reference_generator.gradient_clipping_threshold
      for p in θ
        ∇[p] .*= reference_generator.gradient_clipping_threshold / v
      end
    end
  end

  ∇
end

# this updates the parameters of the NN model
# specifically the optimizer that we have chosen, the Gradient Descent optimizer
function update_parameters!(g, ∇; noise=nothing, rng=nothing, kwargs...)
  θ = Flux.params(g)
  preprocess_gradients!(∇, g, θ; kwargs...)
  # update the neural model
  Optimise.update!(g.optimizer, θ, ∇)

  if !isnothing(noise)
    for p in θ
      p .+= randn(rng, size(p)) * noise
    end
  end
  nothing
end


# wrapper to call actual updater
function update_state_value_predictor!(solver, state, game_value_per_player)
  # create a buffer to store values
  push!(
    solver.state_value_predictor.replay_buffer,
    (; value_target_per_player=game_value_per_player, state),
  )

  if length(solver.state_value_predictor.replay_buffer) >= solver.state_value_predictor.batch_size
    fit_value_predictor!(solver.state_value_predictor)
    empty!(solver.state_value_predictor.replay_buffer)
  end
end

# never called because there is no state value predictor
function fit_value_predictor!(state_value_predictor::NeuralStateValuePredictor)
  # update only if there is something to update
  @assert length(state_value_predictor.replay_buffer) > 0

  for _ in 1:(state_value_predictor.n_epochs_per_update)
    θ = Flux.params(state_value_predictor)
    # the function here is a simple (x - target)^2 / len(replay_buffer)
    ∇L = Zygote.gradient(θ) do
      sum(state_value_predictor.replay_buffer) do d
        sum(v -> v^2, d.value_target_per_player - state_value_predictor(d.state))
      end / length(state_value_predictor.replay_buffer)
    end
    update_parameters!(state_value_predictor, ∇L)
  end
end


# main function
using Flux: Flux
using Zygote: Zygote
using TensorGames: TensorGames
using ForwardDiff: ForwardDiff
using Makie: Makie # this is required to prevent reshape errors. I don't know why


function (solver::LiftedTrajectoryGameSolver)(
  game::TrajectoryGame{<:ProductDynamics},
  initial_state;
  context_state=Float64[],
  min_action_probability=0.05,
  parameter_noise=0.0,
  scale_action_gradients=true,
)
  # Flux and Zygote are machine learning libraries
  # they help in making the NN model 
  n_players = num_players(game)

  # get gradient only if learning, otherwise nothing
  if !isnothing(solver.enable_learning) && any(solver.enable_learning)
    trainable_parameters = Flux.params(solver.trajectory_reference_generators...)
    forward_pass_result, back = Zygote.pullback(
      () -> forward_pass(;
        solver,
        game,
        initial_state,
        context_state,
        min_action_probability,
      ),
      trainable_parameters,
      # there are no layer states here
    )
    # TODO: read cost_gradients
    ∇L_per_player = cost_gradients(back, solver, game)
  else
    forward_pass_result =
      forward_pass(; solver, game, initial_state, context_state, min_action_probability)
    ∇L_per_player = [nothing for _ in 1:n_players]
  end


  (; loss_per_player, info) = forward_pass_result

  # Update θ_i if learning is enabled for player i
  if !isnothing(solver.enable_learning)
    for (reference_generator, weights, enable_player_learning, ∇L) in zip(
      solver.trajectory_reference_generators,
      info.mixing_strategies,
      solver.enable_learning,
      ∇L_per_player,
    )
      if !enable_player_learning
        continue
      end
      action_gradient_scaling = scale_action_gradients ? 1 ./ weights : ones(size(weights))
      update_parameters!(
        reference_generator,
        ∇L;
        noise=parameter_noise,
        solver.rng,
        action_gradient_scaling,
      )
    end
  end

  # currently state value predictor is nothing 
  # so this part is not run
  # TODO: what is the role of state_value_predictor
  if !isnothing(solver.state_value_predictor) &&
     !isnothing(solver.enable_learning) &&
     any(solver.enable_learning)
    update_state_value_predictor!(solver, initial_state, info.game_value_per_player)
  end

  map(
    1:n_players,
    info.mixing_strategies,
    loss_per_player,
    info.trajectories_per_player,
    ∇L_per_player,
  ) do player_i, weights, loss, trajectories, ∇L
    ∇L_norm = if isnothing(∇L)
      0.0
    else
      sum(
        norm(something(∇L[p], 0.0)) for
        p in Flux.params(solver.trajectory_reference_generators...)
      )
    end
    (
      player_i,
      trajectories,
      weights,
      info=(; loss, ∇L_norm),
      solver.rng,
    )
  end
end
