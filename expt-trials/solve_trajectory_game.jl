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


# main struct 
abstract type AbstractStrategy end
Base.@kwdef struct LiftedTrajectoryStrategy{TC,TW,TI,TR} <: AbstractStrategy
    "Player index"
    player_i::Int
    "A vector of actions in continuous domain."
    trajectories::Vector{TC}
    "A collection of weights associated with each candidate action to mix over these."
    weights::TW
    "A dict-like object with additioal information about this strategy."
    info::TI
    "A random number generator to compute pseudo-random actions."
    rng::TR
    "The index of the action that has been sampled when this strategy has been querried for an \
    action the first time (needed for longer open-loop rollouts)"
    action_index::Ref{Int} = Ref(0)
end

struct JointStrategy{T1,T2} <: AbstractStrategy
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

# strategy calls
using StatsBase: Weights, sample
using BlockArrays: Block

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

function (strategy::LiftedTrajectoryStrategy)(state, t)
    if t == 1
        strategy.action_index[] = sample(strategy.rng, Weights(strategy.weights))
    end

    (; xs, us) = strategy.trajectories[strategy.action_index[]]
    if xs[t] != state[Block(strategy.player_i)]
        throw(
        ArgumentError("""
                        This strategy is only valid for states on its trajectory but has been \
                        called for an off trajectory state instead which will likely not \
                        produce meaningful results.
                        """),
        )
    end

    PrecomputedAction(xs[t], us[t], xs[t+1])
end


# main function
using Flux: Flux
using Zygote: Zygote
using TensorGames: TensorGames
using ForwardDiff: ForwardDiff
using Makie: Makie # this is required to prevent reshape errors. I don't know why


function solve_trajectory_game!(
    solver::LiftedTrajectoryGameSolver,
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

    γs = map(
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
        LiftedTrajectoryStrategy(;
            player_i,
            trajectories,
            weights,
            info=(; loss, ∇L_norm, game),
            solver.rng,
        )
    end

    # wrapper for calling both strategies in one go
    JointStrategy(γs)
end
