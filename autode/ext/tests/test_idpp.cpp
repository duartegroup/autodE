#include "idpp.h"
#include <vector>
#include <cmath>
#include <catch2/catch.hpp>

using namespace autode;

TEST_CASE("Image generation from number of atoms and k_spr") {
    Image img(10, 1.0);
    // should create zero arrays of 3 * n_atoms for coordinates and grad
    REQUIRE(img.coords.size() == 30);
    REQUIRE(img.grad.size() == 30);
    // number of atoms and k_spr should always be greater than 0
    REQUIRE_THROWS(Image(0, 1.0));
    REQUIRE_THROWS(Image(10, -1.0));
}

TEST_CASE("IDPP energy and gradient") {
    auto coords_0 = arrx::array1d{{0.0, 0.0, 0.0, 0.5, 0.0, 0.0}};
    auto coords_1 = arrx::array1d{{0.0, 0.0, 0.0, 0.55, 0.0, 0.0}};
    auto coords_2 = arrx::array1d{{0.0, 0.0, 0.0, 0.7, 0.0, 0.0}};
    IDPPPotential pot(coords_0, coords_2, 3);

    auto img_1 = Image(2, 1.0);
    img_1.coords = coords_1;

    pot.calc_idpp_engrad(1, img_1);
    // S =                w ^ -4 * (d_id - d) ^ 2
    // d_id = 0.5 + (0.7 - 0.5) * (1 / 2), d = 0.55
    double known_val = std::pow(0.55, -4) * std::pow((0.5 + 0.2 / 2) - 0.55, 2);

    REQUIRE(std::abs(img_1.en - known_val) < 1e-12);

    auto calc_grad = img_1.grad;
    auto num_grad = arrx::array1d{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};

    constexpr double shift = 1e-9;
    const auto energy = img_1.en;

    for (int i = 0; i < 6; i++) {
        img_1.coords[i] += shift;
        pot.calc_idpp_engrad(1, img_1);
        num_grad[i] = (img_1.en - energy) / shift;
        img_1.coords[i] -= shift;
    }

    for (int i = 0; i < 6; i++) {
        REQUIRE(std::abs(calc_grad[i] - num_grad[i]) < 1e-7);
    }
}
