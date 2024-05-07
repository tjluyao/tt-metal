// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

// #include "debug/dprint.h"  // required in all kernels using DPRINT

void kernel_main() {

    uint32_t in_tile_offset_by_head        = get_arg_val<uint32_t>(0);
    uint32_t q_start_addr                  = get_arg_val<uint32_t>(1);

    constexpr uint32_t ELEMENT_SIZE         = get_compile_time_arg_val(0);
    constexpr uint32_t SUBTILE_LINE_BYTES   = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_q_out          = get_compile_time_arg_val(2);
    constexpr uint32_t head_size            = get_compile_time_arg_val(3);
    constexpr uint32_t batch                = get_compile_time_arg_val(4);
    constexpr uint32_t head_size_num_tiles  = get_compile_time_arg_val(5);
    constexpr uint32_t PHASES_TO_READ       = get_compile_time_arg_val(6);  // 0 to read all phases, 1 to read only first phase, 2 to read only second phase

    constexpr uint32_t num_x                         = get_compile_time_arg_val(7);
    constexpr uint32_t num_y                         = get_compile_time_arg_val(8);
    volatile tt_l1_ptr uint32_t * in0_mcast_noc_x          = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(2));
    volatile tt_l1_ptr uint32_t * in0_mcast_noc_y          = (volatile tt_l1_ptr uint32_t*)(get_arg_addr(2 + num_x));

    // Q
    uint32_t qkv_x = 0;
    uint32_t qkv_y = 0;
    uint32_t total_input_cores = num_x * num_y;
    uint32_t num_tiles_per_core = (head_size_num_tiles * batch) / total_input_cores;

    // // debug for loop
    // DPRINT << "[mikevin DPRINT] NOC coordinates:" << ENDL();
    // DPRINT << "     total_input_cores: " << total_input_cores << ENDL();
    // DPRINT << "     num_tiles_per_core: " << num_tiles_per_core << ENDL();
    // for (uint32_t i = 0; i < num_x; i++){
    //     for (uint32_t j = 0; j < num_y; j++){
    //         DPRINT << "         " << in0_mcast_noc_x[i] << ", " << in0_mcast_noc_y[j] << ENDL();
    //     }
    // }


    uint64_t qkv_read_addr = get_noc_addr(in0_mcast_noc_x[qkv_x], in0_mcast_noc_y[qkv_y], q_start_addr) + in_tile_offset_by_head;
    uint32_t num_tiles_read_cur_core = 0;
    uint32_t q_write_addr = 0;
    uint32_t tile_size = head_size/head_size_num_tiles;
    const uint32_t cb_write_ptr_base = get_write_ptr(cb_id_q_out);

    // DPRINT << "[xuncai DPRINT] head_size = " << head_size << ENDL();
    // DPRINT << "[xuncai DPRINT] head_size_num_tiles = " << head_size_num_tiles << ENDL();
    // DPRINT << "[xuncai DPRINT] ELEMENT_SIZE = " << ELEMENT_SIZE << ENDL();
    // DPRINT << "[xuncai DPRINT] SUBTILE_LINE_BYTES = " << SUBTILE_LINE_BYTES << ENDL();
    // DPRINT << "[xuncai DPRINT] tile_size = " << tile_size << ENDL();
    // DPRINT << "[xuncai DPRINT] batch = " << batch << ENDL();
    // DPRINT << "[xuncai DPRINT] num_kv_heads = " << num_kv_heads << ENDL();
    // DPRINT << "[xuncai DPRINT] in_tile_offset_by_head = " << in_tile_offset_by_head << ENDL();

    // Read 2 phases per tile, where there are batch * q_num_tiles tiles
    // DPRINT << "[xuncai DPRINT] Q read:" << ENDL();
    // DPRINT << "         qkv_read_addr = " << qkv_read_addr << ENDL();
    // DPRINT << "         q_write_addr = " << q_write_addr << ENDL();
    for (uint32_t q = 0; q < batch; ++q) {
        // DPRINT << "[xuncai DPRINT] q = " << q << ENDL();
        uint32_t wptr_offset = q < 16 ? q * SUBTILE_LINE_BYTES : (q - 16) * SUBTILE_LINE_BYTES + 512*ELEMENT_SIZE;
        uint32_t q_write_addr = cb_write_ptr_base + wptr_offset;
        for (uint32_t i = 0; i < head_size_num_tiles; ++i) {
            // DPRINT << "[xuncai DPRINT] i = " << i << ENDL();
            // DPRINT << "         qkv_read_addr = " << qkv_read_addr << ENDL();
            // DPRINT << "         q_write_addr = " << q_write_addr << ENDL();
            // Read first phase
            if constexpr (PHASES_TO_READ == 0 || PHASES_TO_READ == 1) {
                noc_async_read(qkv_read_addr, q_write_addr, SUBTILE_LINE_BYTES);
                //noc_async_read_barrier();
            }
            // Read second phase
            if constexpr (PHASES_TO_READ == 0 || PHASES_TO_READ == 2) {
                noc_async_read(qkv_read_addr+256*ELEMENT_SIZE, q_write_addr+256*ELEMENT_SIZE, SUBTILE_LINE_BYTES);
                //noc_async_read_barrier();
            }
            //noc_async_read_barrier();

            qkv_read_addr += tile_size;
            q_write_addr += tile_size;
            num_tiles_read_cur_core++;

            if (num_tiles_read_cur_core == num_tiles_per_core) {
                qkv_x++;
                if (qkv_x == num_x) {
                    qkv_x = 0;
                    qkv_y++;
                }
                qkv_read_addr = get_noc_addr(in0_mcast_noc_x[qkv_x], in0_mcast_noc_y[qkv_y], q_start_addr) + in_tile_offset_by_head;
                num_tiles_read_cur_core = 0;
            }
        }
    }

    noc_async_read_barrier();
}