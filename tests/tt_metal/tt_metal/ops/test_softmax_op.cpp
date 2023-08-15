#include "tt_metal/host_api.hpp"
#include "libs/tensor/tensor.hpp"
#include "libs/tt_dnn/op_library/softmax/softmax_op.hpp"
#include <tt_numpy/functions.hpp>

#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/llrt/tt_debug_print_server.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace constants;

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    bool pass = true;

    ////////////////////////////////////////////////////////////////////////////
    //                      Initial Runtime Args Parse
    ////////////////////////////////////////////////////////////////////////////
    std::vector<std::string> input_args(argv, argv + argc);
    string arch_name = "";
    try {
        std::tie(arch_name, input_args) =
            test_args::get_command_option_and_remaining_args(input_args, "--arch", "grayskull");
    } catch (const std::exception& e) {
        log_fatal(tt::LogTest, "Command line arguments found exception", e.what());
    }
    const tt::ARCH arch = tt::get_arch_from_string(arch_name);
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    int pci_express_slot = 0;
    tt_metal::Device *device =
        tt_metal::CreateDevice(arch, pci_express_slot);
    pass &= InitializeDevice(device);
    tt_start_debug_print_server(device->cluster(), {0}, {{1, 1}});
    Shape shape = {1, 1, TILE_HEIGHT, TILE_WIDTH};
    Tensor a = tt::numpy::random::random(shape).to(Layout::TILE).to(device);
    Tensor c = tt::operations::primary::softmax_in_place(a);
    Tensor d = c.cpu();
    Tensor host_a = a.cpu(); // Move tensor a to host to validate
    pass &= CloseDevice(device);


    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        log_fatal(LogTest, "Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
