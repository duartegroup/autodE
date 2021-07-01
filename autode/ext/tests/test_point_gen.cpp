#include "points.h"
#include <cmath>
#include <vector>
#include <catch2/catch.hpp>

using namespace std;
using namespace autode;


TEST_CASE("Test point generation with only a single point"){

    REQUIRE_THROWS( 
                   CubePointGenerator(1,  // Single point
                                      1,
                                      0.0,
                                      1.0)
    ); 
}


TEST_CASE("Test the gradient used to minimise the points"){
   
    CubePointGenerator generator(2,  // 2 points in a 1D box with
                                 1,  // a length of unity
                                 0.0,
                                 1.0);

    REQUIRE(generator.half_box_length == Approx(0.5)); 

    generator.n = 1;                 // Override the number of points..
    vector<double> point(1, 0.5);
    generator.points = {point};

    // Check that for a single point the gradient is zero
    generator.set_grad();
    REQUIRE(generator.norm_grad() == Approx(0.0).epsilon(1E-4));

}



TEST_CASE("Test periodic point generation in 1D"){
    CubePointGenerator pointGenerator(2,    // 2 points
                                      1,    // 1D
                                      0.0,  // minimum value
                                      1.0); // maximum value


                               
    pointGenerator.run(1E-3, // Gradient tolerance 
                       0.01, // Step size
                       200); // Maximum number of iterations

    auto points = pointGenerator.points;

    REQUIRE(points.size() == 2);
    REQUIRE(points[0].size() == 1);
   
    // ∆r between the two points should be 0.5 in a periodic 1D system with length of 1 
    REQUIRE(fabs(points[0][0] - points[1][0]) == Approx(0.5).epsilon(0.05));
}


double distance(vector<double> point1, vector<double> point2){
    /* Calculate the Euclidean distance between two points
     * NOTE: Does not consider periodic boundary conditions
    */

    double dist_sq = 0.0;
    int dim = point1.size();
 
    for (int i = 0; i < dim; i++){
        double tmp = point1[i] - point2[i];
        dist_sq += tmp * tmp;
    } // i 
 
    return sqrt(dist_sq);
}


TEST_CASE("Test periodic point generation in 2D square"){
    CubePointGenerator pointGenerator(2,    // 2 points
                                      2,   // 2D  (cube)
                                      0.0,   // unit length
                                      1.0);



    pointGenerator.run(1E-4, 0.01, 100);
    auto points = pointGenerator.points;

    // Minimum ∆r between the two points should be 0.707
    REQUIRE(distance(points[0], points[1]) > 0.6);
}


TEST_CASE("Test periodic point generation in 3D cube"){
    CubePointGenerator pointGenerator(4,    // 4 points
                                      3,   // 3D  (cube)
                                      0.0,   // unit length
                                      1.0);


                               
    pointGenerator.run(1E-4, 0.01, 100);
    auto points = pointGenerator.points;

    // Minimum ∆r between two points needs to be at least 0.4, ideally
    // it would be ~0.7 (perhaps)
    for (int i=0; i < 4; i++){
        for (int j=0; j < i; j++){
            REQUIRE(distance(points[i], points[j]) > 0.4);
        }// j
    }// i
}
