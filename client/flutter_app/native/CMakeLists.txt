cmake_minimum_required(VERSION 3.10.2)
project(federative_har C)

add_library(
    federative_har
    SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../core/federative_har.c
)

target_include_directories(
    federative_har
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../core
)

target_compile_options(
    federative_har
    PRIVATE
    -O3
    -std=c99
    -Wall
    -Wextra
)

target_link_libraries(federative_har m)
