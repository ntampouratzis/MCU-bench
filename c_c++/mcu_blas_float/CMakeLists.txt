add_executable(hello_multicore
        multicore.c
        )

# Add pico_multicore which is required for multicore functionality
target_link_libraries(hello_multicore
        pico_stdlib
        pico_multicore)


# Optimization flags
target_compile_options(hello_multicore PRIVATE
    -O3
    -funroll-loops
    -fomit-frame-pointer
)

pico_enable_stdio_usb(hello_multicore 1)
pico_enable_stdio_uart(hello_multicore 0)

# create map/bin/hex file etc.
pico_add_extra_outputs(hello_multicore)

# add url via pico_set_program_url
example_auto_set_url(hello_multicore)
