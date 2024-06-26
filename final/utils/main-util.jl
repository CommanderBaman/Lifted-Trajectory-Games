using BlockArrays: mortar

# chooser function 
function get_initial_state(num_players::Int, initial_state_option::String="")
  if initial_state_option == "random" || initial_state_option == ""
    # completely random initial state
    initial_state = (mortar([rand(4) for _ in 1:num_players]) .- 0.5) * 4
  elseif initial_state_option == "static"
    # no initial velocity 
    initial_state = (mortar([vcat(rand(2), [0.5, 0.5]) for _ in 1:num_players]) .- 0.5) * 4
  else
    throw("Unimplemented Initial State type")
  end
  initial_state
end