add_executable(test_mpmi_cq1 test_mpmi_cq1.cpp)
add_executable(test_hnsw_reorder_micro test_hnsw_reorder_micro.cpp)
add_executable(test_mpmi_correctness test_mpmi_correctness.cpp)
add_executable(test_knowhere_mpmi_rorder test_knowhere_mpmi_rorder.cpp)

foreach(target IN ITEMS
        test_mpmi_cq1
        test_hnsw_reorder_micro
        test_mpmi_correctness
        test_knowhere_mpmi_rorder)
  set_target_properties(${target} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
  target_link_libraries(${target} PRIVATE faiss)
  if(__RISCV64)
    target_compile_options(${target} PRIVATE -march=rv64gcv_zvfhmin -mabi=lp64d)
  endif()
endforeach()

# The unified driver calls KnowhereConfig::SetSimdType so its S/V/M/F labels
# describe the backend actually selected by the same dispatch layer as Milvus.
target_link_libraries(test_mpmi_cq1 PRIVATE knowhere)
target_link_libraries(test_knowhere_mpmi_rorder PRIVATE knowhere)

# Keep the executable-side LMUL report identical to the definitions used to
# compile the RVV codec.  These cache variables are set by libfaiss.cmake on
# RISC-V and are intentionally absent on other architectures.
if(DEFINED RVANNS_RVV_DECODE_LMUL)
  target_compile_definitions(test_mpmi_cq1 PRIVATE
    RVANNS_RVV_DECODE_LMUL=${RVANNS_RVV_DECODE_LMUL})
endif()
if(DEFINED RVANNS_RVV_ACC_LMUL)
  target_compile_definitions(test_mpmi_cq1 PRIVATE
    RVANNS_RVV_ACC_LMUL=${RVANNS_RVV_ACC_LMUL})
endif()
if(DEFINED RVANNS_RVV_LMUL_TAG)
  target_compile_definitions(test_mpmi_cq1 PRIVATE
    RVANNS_RVV_LMUL_TAG=${RVANNS_RVV_LMUL_TAG})
endif()

add_test(NAME river_mpmi_correctness COMMAND test_mpmi_correctness)
add_test(NAME river_knowhere_mpmi_rorder COMMAND test_knowhere_mpmi_rorder)
add_test(
  NAME river_hnsw_layout_contract
  COMMAND test_hnsw_reorder_micro --rounds 1)
add_test(
  NAME river_hnsw_layout_singleton
  COMMAND test_hnsw_reorder_micro
          --nb 1 --nq 1 --dim 3 --M 2 --efC 4 --efS 2 --k 1 --rounds 1)
set_tests_properties(
  river_mpmi_correctness river_knowhere_mpmi_rorder
  river_hnsw_layout_contract river_hnsw_layout_singleton
  PROPERTIES TIMEOUT 300)
