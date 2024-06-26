


# import all games  
include("two-player-meta-tag.jl")
include("three-coop.jl")

# chooser function 
function form_game(game_type::String, player_bounds::Any)
  if game_type == "2-tag"
    game = two_player_meta_tag(player_bounds=player_bounds)
  elseif startswith(game_type, "3-coop")
    if endswith(game_type, "adv1")
      game = three_player_coop_adv1(player_bounds=player_bounds)
    elseif endswith(game_type, "adv2")
      game = three_player_coop_adv2(player_bounds=player_bounds)
    else
      game = three_player_coop_basic(player_bounds=player_bounds)
    end
  else
    throw("Unimplemented Game Type")
  end


  game
end