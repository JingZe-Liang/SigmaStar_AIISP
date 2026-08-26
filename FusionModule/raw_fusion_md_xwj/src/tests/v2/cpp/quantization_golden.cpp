#include <cfenv>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    if (argc == 1) {
        return 0;
    }
    if (argc < 3 || ((argc - 1) % 2) != 0) {
        return 2;
    }
    if (std::fesetround(FE_TONEAREST) != 0) {
        return 3;
    }
    for (int index = 1; index < argc; index += 2) {
        const float denoised = std::stof(std::string(argv[index]));
        const float injection = std::stof(std::string(argv[index + 1]));
        volatile float scaled = injection * 3795.0f;
        const float value = denoised + scaled;
        const auto rounded = static_cast<std::uint16_t>(std::nearbyint(value));
        std::cout << rounded << '\n';
    }
    return 0;
}
