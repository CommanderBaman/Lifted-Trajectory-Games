# all the reducers for cost

# game value of pursuer
function two_tag_reducer(cost)
  cost[1]
end

# value of evader 
function three_coop_reducer(cost)
  abs(cost[3])
end


# game value of pursuer
function herd_reducer(cost)
  cost[1]
end

# game value of pursuer
function coop_herd_reducer(cost)
  (cost[1] + cost[2]) / 2
end


# chooser function 
function form_reducer(game_type::String)
  if game_type == "2-tag"
    reducer = two_tag_reducer
  elseif startswith(game_type, "3-coop")
    reducer = three_coop_reducer
  elseif contains(game_type, "coop-herd")
    reducer = coop_herd_reducer
  elseif contains(game_type, "herd")
    reducer = herd_reducer
  else
    throw("Unimplemented Reducer for given game type")
  end

  reducer
end