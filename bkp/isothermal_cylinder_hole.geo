//

lcar0 = 0.00015;
lcar1 = 0.0020;
lcar2 = 0.02;


Point(2)  = { -0.003,   0.0, 0.0, lcar0};
Point(3)  = { -0.0015,   0.0, 0.0, lcar0};
Point(4)  = {  0.0,   0.0, 0.0, lcar0};
Point(5)  = { 0.0015,   0.0, 0.0, lcar0};
Point(6)  = { 0.003,   0.0, 0.0, lcar0};
Point(7)  = {-0.150, -0.150, 0.0, lcar2};
Point(8)  = { 0.3, -0.150, 0.0, lcar2};
Point(9)  = { 0.3,  0.150, 0.0, lcar2};
Point(10) = {-0.150,  0.150, 0.0, lcar2};

Point(11) = {-0.009,  0.009, 0.0, lcar1};
Point(12) = { 0.03,  0.009, 0.0, lcar1};
Point(13) = { 0.03, -0.009, 0.0, lcar1};
Point(14) = {-0.009, -0.009, 0.0, lcar1};

//Define bounding box edges
Line(1) = {7, 8};
Line(2) = {8, 9};
Line(3) = {9,10};
Line(4) = {10,7};

Line(5) = {2, 3};
Circle(6) = {3, 4, 5};
Line(7) = {5, 6};
Circle(8) = {5, 4, 3};
Circle(9) = {2, 4, 6};
Circle(10) = {6, 4, 2};

Line(11) = {11, 12};
Line(12) = {12, 13};
Line(13) = {13, 14};
Line(14) = {14, 11};



Transfinite Line { 6} = 26 Using Progression 1.0;
Transfinite Line { 8} = 26 Using Progression 1.0;
Transfinite Line { 9} = 26 Using Progression 1.0;
Transfinite Line {10} = 26 Using Progression 1.0;
Transfinite Line { 5} = 11 Using Progression 1.0;
Transfinite Line { 7} = 11 Using Progression 1.0;


Line Loop(101) = {1,2,3,4,11,12,13,14};
Line Loop(102) = {9,10,11,12,13,14};
Line Loop(103) = {5,6,7,-9};
Line Loop(104) = {-8,5,10,7};


//Define unstructured far field mesh zone
Plane Surface(201) = {-101};
Plane Surface(202) = {102};

Plane Surface(203) = {103};
Transfinite Surface{203} Alternate;

Plane Surface(204) = {-104};
Transfinite Surface{204} Alternate;

Physical Line("inflow") = {4};
Physical Line("surface") = {6,8};
Physical Line("outflow") = {2};
Physical Line("sidewall")={1,3};
Physical Surface("fluid") = {201,202,203,204};

