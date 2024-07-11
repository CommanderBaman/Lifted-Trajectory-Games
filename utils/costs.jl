using Statistics: std, mean
using GLMakie

# I am thinking like this
# the data I have is x = [block(x)(1), ..., block(x)(2)]
# and same for u 
# then I calculate the costs from them
# this is already implemented in the game object
# I am just writing my own functions to simplify it 


# game stage cost - cost at a single interval
function get_game_stage_cost(game, x, u)
  game.cost.stage_cost(x, u,)
end

# cost for a planning horizon
function get_game_cost(game, x, u, reducer_function, planning_horizon=20, context_state=[])
  num_steps = length(x)
  costs = map(1:(num_steps-planning_horizon)) do i
    xs = x[i:i+planning_horizon]
    us = map(ui -> ui.reference_control, u[i:i+planning_horizon])
    reducer_function(game.cost(xs, us, context_state))
  end
  costs
end

# get all costs at a step
function get_cost_vertical(game_costs_array, time)
  map(game_costs_array) do cost_array
    cost_array[time]
  end
end

function get_plot_values(game_costs_array)
  number_of_simulations = length(game_costs_array)
  number_of_steps = length(game_costs_array[1])

  upper_values = []
  mean_values = []
  lower_values = []

  map(1:number_of_steps) do i
    costs = get_cost_vertical(game_costs_array, i)
    mean_value = mean(costs)
    std_value = std(costs)

    push!(mean_values, mean_value)
    push!(upper_values, mean_value + std_value)
    push!(lower_values, mean_value - std_value)
  end

  # lower_values = reshape(lower_values, length(lower_values))
  # upper_values = reshape(upper_values, length(upper_values))
  # mean_values = reshape(mean_values, length(mean_values), 1)
  upper_values, mean_values, lower_values
end

function get_plot_print_values(game_costs_array; length_to_check = 50, round_digits = 3)
  while length_to_check >= length(game_costs_array[1])
    length_to_check -= 10
  end

  upper_values, mean_values, lower_values = get_plot_values(game_costs_array)
  stddevs = upper_values .- mean_values

  # clipping
  mean_values = mean_values[end-length_to_check:end]
  stddevs = stddevs[end-length_to_check:end]
  
  round(mean(mean_values), digits=round_digits), round(mean(stddevs), digits=round_digits)
end


function Makie.convert_arguments(::Type{Plot{Makie.band}}, x::UnitRange{Int64}, ylower::Vector{Any}, yupper::Vector{Any})
  return (Makie.Point2{Float64}.(x, ylower), Makie.Point2{Float64}.(x, yupper))
end


function save_cost_plot(
  game_costs_array;
  fig=Makie.Figure(),
  ax_kwargs=(;),
  aspect=1,
  heading="Cost Plot",
  filename="cost-plot",
)
  number_of_simulations = length(game_costs_array)
  number_of_steps = length(game_costs_array[1])

  upper_values, mean_values, lower_values = get_plot_values(game_costs_array)

  xs = range(1, number_of_steps)



  xlims = (1, number_of_steps)
  ylims = (0, ceil(max(upper_values...) * 1.1))
  title = "$heading"
  ax = Makie.Axis(
    fig[1, 1];
    title,
    aspect,
    limits=(xlims, ylims),
    xlabel="Steps",
    ylabel="Game Value",
    xlabelpadding=0,
    ylabelpadding=-5,
    ax_kwargs...,
  )
  Makie.lines!(xs, mean_values)
  Makie.band!(xs, lower_values, upper_values, color=(:blue, 0.5))

  Makie.save("$filename.png", fig)
end


function generate_random_costs(number_of_simulations=2, number_of_steps=3)
  costs = []
  for i in 1:number_of_simulations
    push!(costs, rand(number_of_steps))
  end
  costs
end