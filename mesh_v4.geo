//

lcar0 = 0.003125;
lcar1 = 0.025;


Point(1)  = { +0.0, 0.0, 0.0, lcar0*5};
Point(2)  = { +0.3, 0.0, 0.0, lcar0};
Point(3)  = { +0.6, 0.0, 0.0, lcar0*5};
Point(4)  = { +0.6, 0.2, 0.0, lcar0*5};
Point(5)  = {  3.0, 0.2, 0.0, lcar1};
Point(6)  = {  3.0, 1.0, 0.0, lcar1};
Point(7)  = {  0.6, 1.0, 0.0, lcar0};
Point(8)  = {  0.0, 1.0, 0.0, lcar1};

//Define bounding box edges
Line( 1) = {1, 2};
Line( 2) = {2, 3};
Line( 3) = {3, 4};
Line( 4) = {4, 5};
Line( 5) = {5, 6};
Line( 6) = {6, 7};
Line( 7) = {7, 8};
Line( 8) = {8, 1};



Line Loop(1) = {1:8};
Plane Surface(1) = {-1};
/*Transfinite Surface {1};*/


Point(10)  = {  0.30, 0.2, 0.0, lcar0};
Point(11)  = {  0.33, 0.3, 0.0, lcar0};
Point(12)  = {  0.35, 0.4, 0.0, lcar0};
Point(13)  = {  0.40, 0.5, 0.0, lcar0};
Point(14)  = {  0.45, 0.6, 0.0, lcar0};
Point(15)  = {  0.50, 0.7, 0.0, lcar0};
Point(16)  = {  0.55, 0.8, 0.0, lcar0};
Point{10:16} In Surface{1};

/*dx = 0.000025;*/
/*dy = 0.000125;*/
/*Point(21)  = {  0.007 + dx, 0.00, 0.0, lcar0};*/
/*Point(22)  = {  0.007 - dx, 0.00, 0.0, lcar0};*/
/*Point(23)  = {  0.007     ,  +dy, 0.0, lcar0};*/
/*Point(24)  = {  0.007     ,  -dy, 0.0, lcar0};*/

Physical Line("inlet") = {8};
Physical Line("outlet") = {5};
Physical Line("wall") = {1,2,3,4,6,7};
Physical Surface("domain") = {1};

