#include "points.h"
#include <iostream>

using namespace std;
using namespace autode;

int main() {

    CubePointGenerator pointGenerator(5,
                                      3,
                                      0.0,
                                      1.0);



    pointGenerator.run(1E-4, 0.01, 200);

    for (auto &point : pointGenerator.points){
        for (auto &component : point){
            cout << component << '\t';
        }
        cout << endl;
    }

    return 0;
}
