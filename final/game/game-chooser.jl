


# import all games  
include("two-player-meta-tag.jl")

# chooser function 
function form_game(game_type::String, player_bounds::Any) 
  if game_type == "2-tag"
    game = two_player_meta_tag(player_bounds=player_bounds)
  else
    throw("Unimplemented Game Type")
  end


  game 
end