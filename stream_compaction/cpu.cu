#include "cpu.h"
#include <cstdio>

#include "common.h"

#define SIMULATE_GPU_SCAN 0


namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
#if SIMULATE_GPU_SCAN
            // placeholder
#else
            odata[0] = 0;
            int sum = 0;
            for (int i  = 1; i < n; i++) {
                sum += idata[i-1];
                odata[i] = sum;
            }
#endif
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int p1 = 0, p2 = 0;
            int count = 0;
            for (;p2 < n && p1 < n;)
            {
                if (idata[p2] != 0)
                {
                    odata[p1] = idata[p2];
                    p1++;
                    p2++;
                    count++;
                }
                else
                {
                    p2++;
                }
            }
            timer().endCpuTimer();
            return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            int* valValid = new int[n];
            int* scanResult = new int[n];

            timer().startCpuTimer();

            // mark valid indices and scan
            for (int i = 0; i < n; i++)
            {
                valValid[i] = idata[i] > 0 ? 1 : 0;
            }

            scanResult[0] = 0;
            int sum = 0;
            for (int i = 1; i < n; i++)
            {
                sum += valValid[i - 1];
                scanResult[i] = sum;
            }

            // compact
            int count = 0;
            for (int i = 0; i < n; i++)
            {
                int val = idata[i];
                if (val > 0)
                {
                    int oIdx = scanResult[i];
                    odata[oIdx] = val;
                    count++;
                }
            }

            timer().endCpuTimer();

            delete[] valValid;
            delete[] scanResult;
            return count;
        }
    }
}
