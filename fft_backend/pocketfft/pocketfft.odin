package pocketfft

import "../../config"
import "core:c"

when !config.USE_FFTW {
    when ODIN_OS == .Windows {
        // build.ps1 uses /MT
        foreign import lib {config.POCKETFFT}
    } else when ODIN_OS == .Darwin {
        foreign import lib {config.POCKETFFT, "system:c++"}
    } else {
        foreign import lib {config.POCKETFFT, "system:stdc++"}
    }
    @(default_calling_convention = "c", link_prefix = "omat_pocketfft_")
    foreign lib {
        plan_create :: proc(n: int, shape, stride_in, stride_out: [^]int, first: int, kind: c.int, single, inverse: bool, threads: int) -> rawptr ---
        execute :: proc(p, input, output: rawptr) ---
        destroy :: proc(p: rawptr) ---
    }
}
