include("receding-horizon.jl")
include("open-loop.jl")

# chooser function 
function form_strategy(strategy_option::String="")
  if strategy_option == "receding-horizon" || strategy_option == ""
    chosen_strategy = RecedingHorizonStrategy
  elseif strategy_option == "open-loop"
    chosen_strategy = OpenLoopStrategy
  else
    throw("Unimplemented Strategy type")
  end
  chosen_strategy
end
