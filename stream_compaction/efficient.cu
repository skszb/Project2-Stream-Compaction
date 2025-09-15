#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int n, int stride, int offset, int* data)
        {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (idx >= n) { return; }

            int rIdx = (idx + 1) * stride - 1;
            int lIdx = rIdx - offset;

            data[rIdx] += data[lIdx];
        }

        __global__ void kernDownSweep(int n, int stride, int offset, int* data)
        {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (idx >= n) { return; }

            int rIdx = (idx + 1) * stride - 1;
            int lIdx = rIdx - offset;

            int rVal = data[rIdx];
            int lVal = data[lIdx];

            data[rIdx] = rVal + lVal;
            data[lIdx] = rVal;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int * idata) {

            const int N = 1 << ilog2ceil(n);
            const int threadsPerBlock = 128;

            // create buffers
            int* dev_buf;
            cudaMalloc(&dev_buf, N * sizeof(int));
            cudaMemset(dev_buf, 0, N * sizeof(int));
            cudaMemcpy(dev_buf, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // compute
            const int maxDepth = ilog2(n - 1);
            timer().startGpuTimer();
            // upsweep
            for (int d = 0; d <= maxDepth; d++)
            {
                int offset = 1 << d;
                int stride = 1 << (d + 1);
                int threadCount = N / stride;
                int numBlocks = (threadCount + threadsPerBlock - 1) / threadsPerBlock;
                kernUpSweep<<< numBlocks, threadsPerBlock >>>(threadCount, stride, offset, dev_buf);
            }

            // downsweep
            cudaMemset(&dev_buf[N - 1], 0, sizeof(int)); 
            for (int d = maxDepth; d >= 0; d--)
            {
                int offset = 1 << d;
                int stride = 1 << (d + 1);
                int threadCount = N / stride;
                int numBlocks = (threadCount + threadsPerBlock - 1) / threadsPerBlock;
                kernDownSweep<<< numBlocks, threadsPerBlock >>>(threadCount, stride, offset, dev_buf);
            }
            timer().endGpuTimer();

            // copy to host and free buffers
            cudaMemcpy(odata, dev_buf, n * sizeof(int), cudaMemcpyDeviceToHost); 
            cudaFree(dev_buf);

            cudaDeviceSynchronize();
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */

        int compact(int n, int *odata, const int *idata) {
            const int N = 1 << ilog2ceil(n);
            const int threadsPerBlock = 128;

            int* dev_idata;
            cudaMalloc(&dev_idata, n * sizeof(int));
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int* dev_odata;
            cudaMalloc(&dev_odata, n * sizeof(int));

            int* dev_bool;
            cudaMalloc(&dev_bool, n * sizeof(int));

            int* dev_scanBuf;
            cudaMalloc(&dev_scanBuf, N * sizeof(int));
            cudaMemset(dev_scanBuf, 0, N * sizeof(int));

            timer().startGpuTimer();
            // calculate mask
            int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
            Common::kernMapToBoolean << <numBlocks, threadsPerBlock >> > (n, dev_bool, dev_idata);
            cudaMemcpy(dev_scanBuf, dev_bool, n * sizeof(int), cudaMemcpyDeviceToDevice);

            // scan
            const int maxDepth = ilog2(n - 1);
            for (int d = 0; d <= maxDepth; d++)
            {
                int offset = 1 << d;
                int stride = 1 << (d + 1);
                int threadCount = N / stride;
                int numBlocks = (threadCount + threadsPerBlock - 1) / threadsPerBlock;
                kernUpSweep << < numBlocks, threadsPerBlock >> > (threadCount, stride, offset, dev_scanBuf);
            }

            cudaMemset(&dev_scanBuf[N - 1], 0, sizeof(int));
            for (int d = maxDepth; d >= 0; d--)
            {
                int offset = 1 << d;
                int stride = 1 << (d + 1);
                int threadCount = N / stride;
                int numBlocks = (threadCount + threadsPerBlock - 1) / threadsPerBlock;
                kernDownSweep << < numBlocks, threadsPerBlock >> > (threadCount, stride, offset, dev_scanBuf);
            }

            // scatter
            Common::kernScatter<<<numBlocks , threadsPerBlock>>>(n, dev_odata, dev_idata, dev_bool, dev_scanBuf);
            timer().endGpuTimer();

            // copy to host and free buffers
            int lastExclusiveResult, lastIsNotZero;
            cudaMemcpy(&lastExclusiveResult, dev_scanBuf + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastIsNotZero, dev_bool + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            int compactElementCount = lastExclusiveResult + lastIsNotZero;

            cudaMemcpy(odata, dev_odata, compactElementCount * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_bool);
            cudaFree(dev_scanBuf);

            cudaDeviceSynchronize();
            return compactElementCount;
        }
    }
}
