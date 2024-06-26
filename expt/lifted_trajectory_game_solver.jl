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
struct LiftedTrajectoryGameSolver{T1,T2,T3,T4,T5,T6,T7,T8,T9,T10}
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

function LiftedTrajectoryGameSolver(
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