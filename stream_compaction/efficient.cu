#include "common.h"
#include "efficient.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>


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
            constexpr int threadsPerBlock = 256;

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
                int threadCount = N >> (d + 1); // N / stride;
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

            checkCUDAError("cuda error");
            return compactElementCount;
        }
    }

    namespace EfficientPlus
    {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernScanBlocksInclusive(int maxDepth, int* blockIncrementData, int* idata)
        {
            int gidx = threadIdx.x + blockIdx.x * blockDim.x;
            int idx = threadIdx.x;
            int blockSize = blockDim.x;

            extern __shared__ int buf[];

            // load input data into shared memory
            int globalIData = idata[gidx];
            buf[idx] = globalIData;

            // upsweep
            for (int d = 0; d <= maxDepth; d++)
            {
                int nodeCount = blockSize >> (d + 1);

                __syncthreads();
                if (idx < nodeCount)
                {
                    int offset = 1 << d;
                    int stride = 1 << (d + 1);

                    int rIdx = (idx + 1) * stride - 1;
                    int lIdx = rIdx - offset;

                    buf[rIdx] += buf[lIdx];
                }
            }

            // down sweep
            if (idx == 0)
            {
                buf[blockSize - 1] = 0;
            }
            for (int d = maxDepth; d >= 0; d--)
            {
                
                int nodeCount = blockSize >> (d + 1);
                __syncthreads();
                if (idx < nodeCount)
                {
                    int offset = 1 << d;
                    int stride = 1 << (d + 1);

                    int rIdx = (idx + 1) * stride - 1;
                    int lIdx = rIdx - offset;

                    int rVal = buf[rIdx];
                    int lVal = buf[lIdx];
                    buf[rIdx] = rVal + lVal;
                    buf[lIdx] = rVal;
                }
            }

            // copy back
            __syncthreads();
            int inclusiveScanData = buf[idx] + globalIData;
            idata[gidx] = inclusiveScanData;
            if (blockIncrementData != nullptr && idx == blockSize - 1)
            {
                blockIncrementData[blockIdx.x] = inclusiveScanData;
            }
        } 

        __global__ void kernAddBlockIncrement(int* idata, const int* incrementData)
        {
            if (blockIdx.x == 0) { return; }
            int idx = threadIdx.x + (blockIdx.x) * blockDim.x;
            // shift left by one block because of incrementData is the result of inclusive scan
            int blockIncrement = incrementData[blockIdx.x - 1];
            idata[idx] += blockIncrement;
        }

        /**
         * work-efficient scan of arbitrary size using shared memory
         */
        void scan(int n, int* odata, const int* idata)
        {
            constexpr int MAX_THREADS_PER_BLOCK = 256;

            // Calculate the parameters and create buffers for each level beforehand, we may need multiple levels of inter block synchronization,
            std::vector<int*> dev_buffers;          // dev_buffers stores the buffers for each level (both input and output since the algorithm is in-place)
            std::vector<int> blockCounts;           // number of blocks for each level
            std::vector<int> perBlockThreadCounts;  // number of threads per block for each level (in the last level we might launch blocks with fewer threads)
            std::vector<int> perBlockMaxDepths;     // the max depth of the up-sweep/down-sweep within each block for each level

            for (int elementCount = n; elementCount > 1;)
            {
                int blockCount = divUp(elementCount, MAX_THREADS_PER_BLOCK);
                int perBlockThreadCount = std::min(MAX_THREADS_PER_BLOCK, elementCount);
                int perBlockMaxDepth = ilog2(perBlockThreadCount - 1);
                int* dev_buf;
                cudaMalloc(&dev_buf, blockCount * perBlockThreadCount * sizeof(int));
                cudaMemset(dev_buf, 0, blockCount * perBlockThreadCount * sizeof(int));

                blockCounts.push_back(blockCount);
                perBlockThreadCounts.push_back(perBlockThreadCount);
                perBlockMaxDepths.push_back(perBlockMaxDepth);
                dev_buffers.push_back(dev_buf);

                // In the next pass we will use the per-block tails as scan input
                elementCount = blockCount;
            }
            cudaMemcpy(dev_buffers[0], idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // scan
            timer().startGpuTimer();
            for (int i = 0; i < dev_buffers.size(); i++)
            {
                // kernel dimension
                int blockCount = blockCounts[i];
                int perBlockThreadCount = perBlockThreadCounts[i];

                // parameters
                int perBlockMaxDepth = perBlockMaxDepths[i];
                int* dev_scanInputBuffer = dev_buffers[i];
                int* dev_blockIncrementBuffer = (i + 1) < dev_buffers.size() ? dev_buffers[i + 1] : nullptr;

                kernScanBlocksInclusive<<<blockCount, perBlockThreadCount, perBlockThreadCount * sizeof(int)>>>(perBlockMaxDepth, dev_blockIncrementBuffer, dev_scanInputBuffer);
                
                // std::string errorMsg = "kernScanBlocksInclusive failed at level " + std::to_string(i) + "\n";
                // cudaDeviceSynchronize();
                // checkCUDAError(errorMsg.c_str());
            }

            for (int i = dev_buffers.size() - 1; i > 0; i--)
            {
                // kernel dimension
                int blockCount = blockCounts[i - 1];
                int perBlockThreadCount = perBlockThreadCounts[i - 1];

                // parameters
                int* dev_blockIncrementBuffer = dev_buffers[i];
                int* dev_scanInputBuffer = dev_buffers[i - 1];

                kernAddBlockIncrement <<< blockCount , perBlockThreadCount >>> (dev_scanInputBuffer, dev_blockIncrementBuffer);

                // std::string errorMsg = "kernAddBlockIncrement failed at level " + std::to_string(i) + "\n";
                // cudaDeviceSynchronize();
                // checkCUDAError(errorMsg.c_str());
            }
            timer().endGpuTimer();

            odata[0] = 0;
            cudaMemcpy(odata + 1, dev_buffers[0], (n-1) * sizeof(int), cudaMemcpyDeviceToHost);

            // free buffers
            for (int* buf : dev_buffers)
            {
                cudaFree(buf);
            }
            checkCUDAError("");
        }
    }
}
