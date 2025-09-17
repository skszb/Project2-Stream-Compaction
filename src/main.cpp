/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include "testing_helpers.hpp"

const int SIZE = 1 << 26; // feel free to change the size of array
const int NPOT = SIZE - 3; // Non-Power-Of-Two
int *a = new int[SIZE];
int *cpuResult = new int[SIZE];
int *c = new int[SIZE];

#define PROFILING 1

#define TEST_ITERATIONS 8

#define SCAN_TEST           1
#define SCAN_CPU            1
#define SCAN_NAIVE          1
#define SCAN_EFFICIENT      1
#define SCAN_EFFICIENT_PLUS 1
#define SCAN_THRUST         1

#define COMPACT_TEST        1


int main(int argc, char* argv[]) {
#if SCAN_TEST
    // Scan tests
    printf("\n");
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");

    genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    // initialize cpuResult using StreamCompaction::CPU::scan you implement
    // We use cpuResult for further comparison. Make sure your StreamCompaction::CPU::scan is correct.
    // At first all cases passed because cpuResult && c are all zeroes.
    float t;

#if SCAN_CPU
    printDesc("cpu scan, power-of-two");
    t = testForIterations(TEST_ITERATIONS, &PerformanceTimer::getCpuElapsedTimeForPreviousOperation, StreamCompaction::CPU::timer(), 
        StreamCompaction::CPU::scan, SIZE, cpuResult, a);
    printElapsedTime(t, "(std::chrono Measured)");
    printArray(SIZE, cpuResult, true);

    zeroArray(SIZE, c);
    printDesc("cpu scan, non-power-of-two");
    t = testForIterations(TEST_ITERATIONS, &PerformanceTimer::getCpuElapsedTimeForPreviousOperation, StreamCompaction::CPU::timer(), 
        StreamCompaction::CPU::scan, NPOT, c, a);
    printElapsedTime(t, "(std::chrono Measured)");
    printArray(NPOT, c, true);
    printCmpResult(NPOT, cpuResult, c);
#endif

#if SCAN_NAIVE
    printDesc("naive scan, power-of-two");
    zeroArray(SIZE, c);
    t = testForIterations(TEST_ITERATIONS, &PerformanceTimer::getGpuElapsedTimeForPreviousOperation, StreamCompaction::Naive::timer(), 
        StreamCompaction::Naive::scan, SIZE, c, a);
    printElapsedTime(t, "(CUDA Measured)");
    // printArray(SIZE, c, true);
    printCmpResult(SIZE, cpuResult, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, non-power-of-two");
    t = testForIterations(TEST_ITERATIONS, &PerformanceTimer::getGpuElapsedTimeForPreviousOperation, StreamCompaction::Naive::timer(), 
        StreamCompaction::Naive::scan, NPOT, c, a);
    printElapsedTime(t, "(CUDA Measured)");
    // printArray(NPOT, c, true);
    printCmpResult(NPOT, cpuResult, c);
#endif

#if SCAN_EFFICIENT
    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two");
    t = testForIterations(TEST_ITERATIONS, &PerformanceTimer::getGpuElapsedTimeForPreviousOperation, StreamCompaction::Efficient::timer(), 
        StreamCompaction::Efficient::scan, SIZE, c, a);
    printElapsedTime(t, "(CUDA Measured)");
    // printArray(SIZE, c, true);
    printCmpResult(SIZE, cpuResult, c);
    
    zeroArray(SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
    t = testForIterations(TEST_ITERATIONS, &PerformanceTimer::getGpuElapsedTimeForPreviousOperation, StreamCompaction::Efficient::timer(), 
        StreamCompaction::Efficient::scan, NPOT, c, a);
    printElapsedTime(t, "(CUDA Measured)");
    printArray(NPOT, c, true);
    printCmpResult(NPOT, cpuResult, c);
#endif

#if SCAN_EFFICIENT_PLUS
    zeroArray(SIZE, c);
    printDesc("work-efficient-plus scan, power-of-two");
    t = testForIterations(TEST_ITERATIONS, &PerformanceTimer::getGpuElapsedTimeForPreviousOperation, StreamCompaction::EfficientPlus::timer(),
        StreamCompaction::EfficientPlus::scan, SIZE, c, a);
    printElapsedTime(t, "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, cpuResult, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient-plus scan, non-power-of-two");
    t = testForIterations(TEST_ITERATIONS, &PerformanceTimer::getGpuElapsedTimeForPreviousOperation, StreamCompaction::EfficientPlus::timer(),
        StreamCompaction::EfficientPlus::scan, NPOT, c, a);
    printElapsedTime(t, "(CUDA Measured)");
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, cpuResult, c);
#endif

#if SCAN_THRUST
    zeroArray(SIZE, c);
    printDesc("thrust scan, power-of-two");
    t = testForIterations(TEST_ITERATIONS, &PerformanceTimer::getGpuElapsedTimeForPreviousOperation, StreamCompaction::Thrust::timer(), 
        StreamCompaction::Thrust::scan, SIZE, c, a);
    printElapsedTime(t, "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, cpuResult, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, non-power-of-two");
    t = testForIterations(TEST_ITERATIONS, &PerformanceTimer::getGpuElapsedTimeForPreviousOperation, StreamCompaction::Thrust::timer(), 
        StreamCompaction::Thrust::scan, NPOT, c, a);
    printElapsedTime(t, "(CUDA Measured)");
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, cpuResult, c);
#endif


#endif

#if COMPACT_TEST
    printf("\n");
    printf("*****************************\n");
    printf("** STREAM COMPACTION TESTS **\n");
    printf("*****************************\n");

    // Compaction tests
    genArray(SIZE - 1, a, 4);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0; 
    int count, expectedCount, expectedNPOT;

    // initialize cpuResult using StreamCompaction::CPU::compactWithoutScan you implement
    // We use cpuResult for further comparison. Make sure your StreamCompaction::CPU::compactWithoutScan is correct.
    zeroArray(SIZE, cpuResult);
    printDesc("cpu compact without scan, power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(SIZE, cpuResult, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedCount = count;
    printArray(count, cpuResult, true);
    printCmpLenResult(count, expectedCount, cpuResult, cpuResult);

    zeroArray(SIZE, c);
    printDesc("cpu compact without scan, non-power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedNPOT = count;
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, cpuResult, c);

    zeroArray(SIZE, c);
    printDesc("cpu compact with scan");
    count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, cpuResult, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, power-of-two");
    count = StreamCompaction::Efficient::compact(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, cpuResult, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
    count = StreamCompaction::Efficient::compact(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, cpuResult, c);
#endif
    fflush(stdout);

#if !PROFILING
    system("pause"); // stop Win32 console from closing on exit
#endif


    delete[] a;
    delete[] cpuResult;
    delete[] c;
}
