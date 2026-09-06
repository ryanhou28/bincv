# Worked examples. Not tests and not benchmarks: they exist to be READ, and to
# prove the kernel set composes into something a real pipeline shape can use
# (T4.3b).
#
# NOT gated on OpenCV. The core-only build is the configuration the whole memory
# argument rests on, and until #30 this directory was skipped there -- so the one
# thing a reader could RUN to check the embedded claim needed the dependency the
# claim is about not needing. vio_frontend now builds in every configuration: its
# OpenCV pieces (cv::imread over a directory, the cv:: sensor arm) sit behind
# BINCV_WITH_OPENCV inside the one file, and the core-only build reads frames
# from a sequence blob (scripts/make_sequence_blob.py) and runs binCV's own
# sensor stage. An example that cannot compile core-clean with its OpenCV half
# guarded off does not belong in this directory.
file(GLOB EXAMPLE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
foreach(example_src ${EXAMPLE_SOURCES})
    get_filename_component(example_name ${example_src} NAME_WE)
    add_executable(${example_name} ${example_src})
    target_link_libraries(${example_name} PRIVATE bincv_core bincv_warnings)
endforeach()
