using Makie: Makie
using GeometryBasics: GeometryBasics

# visualization 

# Makie.plottype(::TrajectoryGamesBase.PolygonEnvironment) = Makie.Poly


function visualize!(
    canvas,
    γ::Makie.Observable{<:LiftedTrajectoryStrategy};
    color = :black,
    weight_offset = 0.0,
)
    trajectory_colors = Makie.@lift([(color, w + weight_offset) for w in $γ.weights])
    Makie.series!(canvas, γ; color = trajectory_colors)
end


function visualize!(
  canvas,
  environment::PolygonEnvironment;
  color = :lightgray,
  kwargs...,
  )
  Makie.poly!(canvas, environment; color, kwargs...)
end

function Makie.convert_arguments(::Type{<:Makie.Poly}, environment::PolygonEnvironment)
  geometry = GeometryBasics.Polygon(GeometryBasics.Point{2}.(environment.set.vertices))
  (geometry,)
end

# called to plot all the points on the trajectory i.e. faded out trajectory
function Makie.convert_arguments(::Type{<:Makie.Series}, γ::LiftedTrajectoryStrategy)
    traj_points = map(γ.trajectories) do traj
        map(s -> Makie.Point2f(s[1:2]), traj.xs)
    end
    (traj_points,)
end

# custom functions
function game_distance_cost(simulation_wrapper)
  state = simulation_wrapper.state
  x1, x2 = blocks(state)
  round(sqrt(0.1 + norm(x1[1:2] - x2[1:2])), digits=3)
end

function game_stage_cost(simulation_wrapper)
  state = simulation_wrapper.state
  control = simulation_wrapper.control.reference_control

  round(game.cost.stage_cost(state, control, [], [])[1], digits=3)
end


# frame forming function
function visualize_sim_step(
  game,
  step;
  fig = Makie.Figure(),
  ax_kwargs = (;),
  xlims = (-5, 5),
  ylims = (-5, 5),
  aspect = 1,
  player_colors = range(colorant"red", colorant"blue", length = num_players(game)),
  player_names = ["Pursuer", "Evader"],
  weight_offset = 0.0,
  heading = "",
  show_legend = false,
  show_turn = false,
  show_costs = false,
)
  s = Makie.Observable(step)

  if !show_turn
      title = "$heading"
  else
      title = Makie.@lift "$heading\nstep: $($s.turn)"
  end

  if show_costs 
    dc = Makie.lift(game_distance_cost, s)
    sc = Makie.lift(game_stage_cost, s)
    title = Makie.@lift "$heading\nstep: $($s.turn) distance: $($dc) stage: $($sc)"
  end



  ax = Makie.Axis(
      fig[1, 1];
      title,
      aspect,
      limits = (xlims, ylims),
      xlabel = "Horizontal position [m]",
      ylabel = "Vertical position [m]",
      xlabelpadding = 0,
      ylabelpadding = -5,
      ax_kwargs...,
  )


  visualize!(ax, game.env)

  plots = []

  for ii in eachindex(s[].strategy.substrategies)
      color = player_colors[ii]
      name = player_names[ii]
      γ = Makie.@lift $s.strategy.substrategies[ii]
      pos = Makie.@lift Makie.Point2f($s.state[Block(ii)][1:2])

      scatter = Makie.scatter!(ax, pos; color, label=name)
      visualize!(ax, γ; weight_offset, color)
      push!(plots, [scatter])
  end

  if show_legend
    Makie.Legend(fig[0,1], plots, player_names, orientation = :horizontal, halign = :left)
  end

  fig, s
end

# main functions 

# called second
function animate_sim_steps(
  game::TrajectoryGame,
  steps;
  filename = "sim_steps",
  heading = filename,
  framerate = 10,
  live = true,
  kwargs...,
)
  # first call to generate the figure object
  fig, s = visualize_sim_step(game, steps[begin]; heading, kwargs...)
  
  # this is where the loop is happening
  # using observables all one has to do is just step and the variables keep auto updating
  Makie.record(fig, "$filename.mp4", steps; framerate) do step
    dt = @elapsed s[] = step
    if live
          time_to_sleep = 1 / framerate - dt
          sleep(max(time_to_sleep, 0.0))
        end
      end
  # commenting the figure out because its no use to see the figure after the video has ended
  # fig
end

# called first 
function animate_sim_steps(game::TrajectoryGame, steps::NamedTuple; kwargs...)
  states, inputs, strategies = steps
  sim_steps =
      map(Iterators.countfrom(), states, inputs, strategies) do turn, state, control, strategy
          (; turn, state, control, strategy)
      end

  animate_sim_steps(game, sim_steps; kwargs...)
end
