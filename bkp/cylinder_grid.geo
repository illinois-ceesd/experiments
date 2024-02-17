//

lcar0 = 0.050;
lcar1 = 0.200;
lcar2 = 2.500;


Point(2)  = { -0.6,   0.0, 0.0, lcar0};
Point(3)  = { -0.5,   0.0, 0.0, lcar0};
Point(4)  = {  0.0,   0.0, 0.0, lcar0};
Point(5)  = {  0.5,   0.0, 0.0, lcar0};
Point(6)  = {  0.6,   0.0, 0.0, lcar0};
Point(7)  = {-25.0, -25.0, 0.0, lcar2};
Point(8)  = { 50.0, -25.0, 0.0, lcar2};
Point(9)  = { 50.0,  25.0, 0.0, lcar2};
Point(10) = {-25.0,  25.0, 0.0, lcar2};

Point(11) = {-1.5,  1.5, 0.0, lcar1};
Point(12) = { 5.0,  1.5, 0.0, lcar1};
Point(13) = { 5.0, -1.5, 0.0, lcar1};
Point(14) = {-1.5, -1.5, 0.0, lcar1};

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

Point(15)  = { -0.4,   0.0, 0.0, lcar0};
Point(16)  = {  0.4,   0.0, 0.0, lcar0};
Line(15) = {3,15};
Circle(16) = {15, 4, 16};
Line(17) = {16,5};
Circle(18) = {16, 4, 15};


Transfinite Line { 6} = 41 Using Progression 1.0;
Transfinite Line { 8} = 41 Using Progression 1.0;
Transfinite Line { 9} = 41 Using Progression 1.0;
Transfinite Line {10} = 41 Using Progression 1.0;
Transfinite Line {16} = 41 Using Progression 1.0;
Transfinite Line {18} = 41 Using Progression 1.0;
Transfinite Line { 5} = 7 Using Progression 1.0;
Transfinite Line { 7} = 7 Using Progression 1.0;
Transfinite Line {15} = 7 Using Progression 1.0;
Transfinite Line {17} = 7 Using Progression 1.0;

Line Loop(101) = {1,2,3,4,11,12,13,14};
Line Loop(102) = {9,10,11,12,13,14};
Line Loop(103) = {5,6,7,-9};
Line Loop(104) = {-8,5,10,7};
Line Loop(105) = {15,-18,17,8};
Line Loop(106) = {15,16,17,-6};
Line Loop(107) = {16,18};

//Define unstructured far field mesh zone
Plane Surface(201) = {-101};
Plane Surface(202) = {102};

Plane Surface(203) = {103};
Transfinite Surface{203} Alternate;

Plane Surface(204) = {-104};
Transfinite Surface{204} Alternate;

Plane Surface(205) = {-105};
Transfinite Surface{205} Alternate;
Plane Surface(206) = { 106};
Transfinite Surface{206} Alternate;
Plane Surface(207) = {-107};

Physical Line("inflow") = {4};
Physical Line("outflow") = {2};
Physical Line("sidewall")={1,3};
Physical Surface("fluid") = {201,202,203,204};
Physical Surface("solid") = {205,206,207};
