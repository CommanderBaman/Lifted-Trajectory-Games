# [(L2 n)/4 tan(180/n)]

# source: https://www.cuemath.com/geometry/area-of-polygons/
# search in faq for area of polygon with n sides
function area_of_polygon(r, n)
  theta = 180/n 
  l = 2 * r * sind(theta)
  area = l^2 * n / 4 / tand(theta)
end

function percent_covered(r, n)
  area_circle = pi * r * r 
  area_polygon = area_of_polygon(r, n)
  round(area_polygon * 100 / area_circle, digits=3)
end

map(3:100) do n
  covered = percent_covered(1, n)
  println("for n = $n: $covered% covered")
end

v = 3 / 2 / pi
println("$v")