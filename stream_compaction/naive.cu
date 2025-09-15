#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernInclusiveScan(int n, int d, int* odata, const int *idata)
        {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if ( idx >= n ) { return; }

            int offset = 1 << (d - 1);
            if ( idx < offset )
            {
                odata[idx] = idata[idx];
            }
            else
            {
                int self = idata[idx];
                int prev = idata[idx - offset];
                int result = self + prev;
                odata[idx] = result;
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata)
        {
            const int maxDepth = ilog2ceil(n);
            const int paddedBufSize = 1 << maxDepth;
            const int threadsPerBlock = 256;
            const int numBlocks = (paddedBufSize + threadsPerBlock - 1) / threadsPerBlock;

            // create buffers
            int* dev_buf1;
            int* dev_buf2;
            cudaMalloc(&dev_buf1, paddedBufSize * sizeof(int));
            cudaMalloc(&dev_buf2, paddedBufSize * sizeof(int));

            cudaMemset(dev_buf1, 0, paddedBufSize);
            cudaMemset(dev_buf2, 0, paddedBufSize);
            cudaMemcpy(dev_buf1, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // computation
            int* pIBuffer = dev_buf1;
            int* pOBuffer = dev_buf2;

            timer().startGpuTimer();
            for (int d = 1; d <= maxDepth; d++)
            {
                kernInclusiveScan<<<numBlocks, threadsPerBlock>>>(paddedBufSize, d, pOBuffer, pIBuffer);
                std::swap(pIBuffer, pOBuffer);
            }
            timer().endGpuTimer();

            pOBuffer = pIBuffer;

            // copy back to host
            odata[0] = 0;
            cudaMemcpy(odata + 1, pOBuffer, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);

            // free buffers
            cudaFree(dev_buf1);
            cudaFree(dev_buf2);

            cudaDeviceSynchronize();
        }
    }
}
