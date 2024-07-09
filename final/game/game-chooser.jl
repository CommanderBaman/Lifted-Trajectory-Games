


# import all games  
include("two-player-meta-tag.jl")
include("three-coop.jl")
include("n-herd.jl")
include("5-herd.jl")
include("3-herd.jl")
include("3-coop-herd.jl")
include("5-coop-herd.jl")

# chooser function 
function form_game(game_type::String, player_bounds::Any)
  if game_type == "2-tag"
    game = two_player_meta_tag(player_bounds=player_bounds)
  elseif contains(game_type, "coop-herd")
    number_of_players = parse(Int, split(game_type, "-")[1])
    @assert number_of_players + 2 == length(player_bounds)

    if number_of_players == 3
      game = herd_coop_3_basic(player_bounds=player_bounds)
    elseif number_of_players == 5
      game = herd_coop_5_basic(player_bounds=player_bounds)
    else
      throw("Unimplemented Game Type")
    end
  elseif startswith(game_type, "3-coop")
    if endswith(game_type, "adv1")
      game = three_player_coop_adv1(player_bounds=player_bounds)
    elseif endswith(game_type, "adv2")
      game = three_player_coop_adv2(player_bounds=player_bounds)
    else
      game = three_player_coop_basic(player_bounds=player_bounds)
    end
  elseif contains(game_type, "herd")
    # guarantee number of players is same as the length of player bounds
    number_of_players = parse(Int, split(game_type, "-")[1])
    @assert number_of_players == length(player_bounds)
    if number_of_players == 3
      if endswith(game_type, "adv1")
        game = herd3_adv1(player_bounds=player_bounds)
      else
        game = herd3_basic(player_bounds = player_bounds)
      end
    elseif number_of_players == 5
      game = herd5_basic(player_bounds = player_bounds)
    else
      game = n_herd_basic(player_bounds = player_bounds)
    end
  else
    throw("Unimplemented Game Type")
  end


  game
end