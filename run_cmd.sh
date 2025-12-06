clear &&                     \
clang++ ./prime.cpp          \
        -O3                  \
        -ffast-math          \
        -march=native        \
        -mtune=native        \
        -flto                \
        -fomit-frame-pointer \
        -funroll-loops       \
        -DNDEBUG             \
        -o prime &&          \
./prime
