l = 5;
lc = l/10;
Point(1) = {-l,-l,0,lc/2};
Point(2) = {l,-l,0,lc/2};
Point(3) = {l,l,0,lc};
Point(4) = {-l,l,0,lc};
Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};
Line Loop(1) = {1,2,3,4};
Plane Surface(1) = {1};

/*
Point(5) = {.0,.0,0};
Field[1] = Distance;
Field[1].NodesList = {5};
Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = lc/4;
Field[2].LcMax = lc;
Field[2].DistMin = l/5;
Field[2].DistMax = l/1;
Background Field = 2;
*/

Physical Line("Wall") = {1,2,3,4};
Physical Surface("Domain") = {1};
Mesh.Algorithm=5;
