#include "points.h"
#include <iostream>
#include <catch2/catch.hpp>

using namespace std;
using namespace autode;


TEST_CASE("Test periodic point generation in 1D"){
    CubePointGenerator pointGenerator(2,
                                      1,
                                      0.0,
                                      1.0);



    pointGenerator.run(1E-4, 0.01, 200);

    for (auto &point : pointGenerator.points){
        for (auto &component : point){
            cout << component << '\t';
        }
        cout << endl;
    }
}
