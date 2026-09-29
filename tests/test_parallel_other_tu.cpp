// The other half of Parallel.BackendInstalledFromAnotherUnitIsVisibleHere.
//
// This unit includes ONLY threads/pool.hpp, the way an integrator's thread-setup
// file would, and installs a pool from here. test_parallel.cpp includes the ops
// headers and checks that the kernels' side sees the same backend. parallel.hpp
// once opened its namespace before the ABI-namespace macro was defined, so a unit
// shaped like this one owned a second set of backend statics and a pool installed
// from it was silently invisible to every kernel -- serial, with nothing to say so.
#include <memory>

#include "bincv/threads/pool.hpp"

namespace bincv_test_other_tu {
namespace {
std::unique_ptr<bincv::ThreadPool> gPool;
}

void installPool(int threads) {
    gPool = std::make_unique<bincv::ThreadPool>(threads);
    gPool->install();
}

void uninstallPool() { gPool.reset(); }

int threadsSeenHere() { return bincv::getNumThreads(); }
} // namespace bincv_test_other_tu
