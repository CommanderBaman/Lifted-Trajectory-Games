
# figuring out the cost of the trajectory references generators 
# TrajectoryReferenceParameterization

# this doesn't work 
# what are these parameters, they directly effect the cost
# now I more or less know how cost is calculated
us = [[1, 2], [3, 4]];
ps = [[5, 6], [7, 8]];
# us = reshape(us, :, horizon) |> eachcol
# ps = reshape(ps, :, horizon) |> eachcol
xs = [[9, 10, 11], [12, 13, 14]];


n = horizon * (state_dim + control_dim)
control_dim = size(first(us), 2);
horizon = 2;
paramter_dim = horizon * control_dim;

# parameterization alpha
a = 3;

cost = sum(zip(xs, us, ps)) do (x, u, param)
    c = sum(0.5 .* a .* u .^ 2 .- u .* param)
    println(u, param, c)
    c
end