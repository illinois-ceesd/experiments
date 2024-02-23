//

lcar0 = 0.00015;
lcar1 = 0.0020;
lcar2 = 0.02;
lcar3 = 0.001;

x=0.1;
y=0.1;
r_glass = 5e-3;
t_glass = 1e-3;
l_glass = 9e-2;

Point(2)  = { -0.003,   0.0, 0.0, lcar0};
Point(3)  = { -0.0015,   0.0, 0.0, lcar0};
Point(4)  = {  0.0,   0.0, 0.0, lcar0};
Point(5)  = { 0.0015,   0.0, 0.0, lcar0};
Point(6)  = { 0.003,   0.0, 0.0, lcar0};
Point(7)  = {-x, -y, 0.0, lcar2};
Point(8)  = { x, -y, 0.0, lcar2};
Point(9)  = { x,  y, 0.0, lcar2};
Point(10) = {-x,  y, 0.0, lcar2};

Point(11) = {-0.009,  0.009, 0.0, lcar1};
Point(12) = { 0.009,  0.009, 0.0, lcar1};
Point(13) = { 0.009, -0.03, 0.0, lcar1};
Point(14) = {-0.009, -0.03, 0.0, lcar1};


//Glass tube points
Point(30) = {r_glass,  y, 0.0, lcar1};
Point(31) = {r_glass,  y-l_glass, 0.0, lcar3};
Point(32) = {r_glass-t_glass,  y-l_glass, 0.0, lcar3};

Point(33) = {r_glass-t_glass,  y, 0.0, lcar1};
Point(34) = {-r_glass+t_glass,  y, 0.0, lcar1};
Point(35) = {-r_glass+t_glass,  y-l_glass, 0.0, lcar3};
Point(36) = {-r_glass,  y-l_glass, 0.0, lcar3};
Point(37) = {-r_glass,  y, 0.0, lcar1};



//Define bounding box edges
Line(1) = {7, 8};
Line(2) = {8, 9};
//Line(3) = {9,10};
Line(4) = {10,7};

Line(5) = {2, 3};
Circle(6) = {3, 4, 5};
Line(7) = {5, 6};
Circle(8) = {5, 4, 3};
Circle(9) = {2, 4, 6};
Circle(10) = {6, 4, 2};

//Line(11) = {11, 12};
//Line(12) = {12, 13};
//Line(13) = {13, 14};
//Line(14) = {14, 11};

//glass tube lines

Line(20) = {9, 30};
Line(21) = {30, 31};
Line(22) = {31, 32};
Line(23) = {32, 33};
Line(24) = {33, 34};
Line(25) = {34, 35};
Line(26) = {35, 36};
Line(27) = {36, 37};
Line(28) = {37, 10};

Transfinite Line { 6} = 26 Using Progression 1.0;
Transfinite Line { 8} = 21 Using Progression 1.0;
Transfinite Line { 9} = 26 Using Progression 1.0;
Transfinite Line {10} = 21 Using Progression 1.0;
Transfinite Line { 5} = 11 Using Progression 1.0;
Transfinite Line { 7} = 11 Using Progression 1.0;


Line Loop(101) = {1,2,20:28,4,9,10};
Line Loop(103) = {5,6,7,-9};
Line Loop(104) = {-8,5,10,7};


//Define unstructured far field mesh zone
Plane Surface(201) = {-101};
Point{11:14} In Surface{201};

Plane Surface(203) = {103};
Transfinite Surface{203} Alternate;

Plane Surface(204) = {-104};
Transfinite Surface{204} Alternate;


Physical Line("inflow") = {24};
Physical Line("glasstube") = {21,22,23,25,26,27};
Physical Line("surface") = {6,8};
Physical Line("outflow") = {1};
Physical Line("sidewall")={4,2,28,20};
Physical Surface("fluid") = {201,203,204};



