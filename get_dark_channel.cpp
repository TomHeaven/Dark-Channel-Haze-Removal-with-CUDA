#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cmath>

//#define DEBUG

/**
 Device code
 */

// Set this const according to b2
#define B2 7
// Set this const according to wnd
#define WND 10
// derived const for vector size in kernel function
#define VEC_SIZE ((WND+WND+1)*(WND+WND+1))

// check error
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false)
{
    if (code != cudaSuccess)
    {
        printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    } else {
        printf("cuda returned code == cudaSuccess\n");
    }
}

template<typename T>
void __device__ swap(T& a, T& b) {
    T t = a;
    a = b;
    b = t;
}


void __device__  adjustTop(double* heap, int* payload,  int sel, int n) {
    if (sel >= n )
        return;
    
    int next = sel + sel;
    if (sel == 0)
        next = 1;
    
    if (next + 1 < n && heap[next + 1] > heap[next])
        ++next;
    
    if (next < n && heap[next] > heap[sel]) {
        swap(heap[next], heap[sel]);
        swap(payload[next], payload[sel]);
    }
    
    adjustTop(heap, payload, next, n);
}

// select top k with maximum root heap
void __device__ selectTopK(const double* d, const int*ind, int n, int k, double* resDis, int* resInd) {
    for(int i = 0; i < k; ++i) {
        resDis[i] = d[i];
        resInd[i] = ind[i];
        adjustTop(resDis, resInd, 0, i);
    }
    
    for(int i = k; i < n; ++i) {
        if (d[i] < resDis[0]) {
            resDis[0] = d[i];
            resInd[0] = ind[i];
            adjustTop(resDis, resInd, 0, k);
        }
    }
}

/**
 % input:
 % startThreadNum, one task may require several launches, this param records the start thread num to continue woring.
 % X, image patches with noise, (b2)^2 * x_height
 % x_height, the height of matrix X
 % N & M, height & width of the original image
 % b2, the size of the image patch
 % wnd, the radius of search window
 % step, step for search
 % num, keep the 'num' most similar patches in each search window
 % d_info, vector for debug
 %
 % output:
 % X_d, denoised image patch, (b2)^2 * x_height
 */
void __global__  denoiseKernel(const int startThreadNum, const double* X, const int x_height, const int x_width, const int N, const int M, const int b2, const int wnd, const int step, const int num, double* X_d, float* d_info)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x + startThreadNum;
    int i = idx / M;
    int j = idx % M;
    
    int off = j*N + i; //zero based
    int rmin = max(i - wnd, 0);
    int rmax = min(i + wnd, N-1);
    int cmin = max(j - wnd, 0);
    int cmax = min(j + wnd, M-1);
    
#ifdef DEBUG
    if (idx == 0) {
        d_info[0] = idx;
        d_info[1] = i;
        d_info[2] = j;
        d_info[3] = off;
        d_info[4] = rmin;
        d_info[5] = rmax;
        d_info[6] = cmin;
        d_info[7] = cmax;
        d_info[8] = N;
        d_info[9] = M;
        d_info[10] = wnd;
    }
#endif
    
    // int vec_size = ceil((rmax - rmin) / step);
    double dis[VEC_SIZE]; //distance vector
    int ind[VEC_SIZE];   // index vector
    double resDis[VEC_SIZE];
    int resInd[VEC_SIZE];
    
    int cnt = 0;
    for(int y = cmin; y <= cmax; y += step) {
        for(int x = rmin; x <= rmax; x += step)
        {
            int off_tmp = y * N + x;
            double sum = 0.0;
            for(int k = 0; k < x_height; ++k) {
                double patch_tmp = X[k * x_width + off_tmp ];
                double patch_ref = X[k * x_width + off ];
                sum += (patch_tmp - patch_ref)*(patch_tmp - patch_ref);
            }
            ind[cnt] = off_tmp;
            dis[cnt++] = sum / x_height;
        }
    }
    
    selectTopK(dis, ind, cnt, num, resDis, resInd);
#ifdef DEBUG
    if (idx == 0) {
        d_info[11] = cnt;
        d_info[12] = dis[10];
        d_info[13] = dis[10];
        
        for(int x = 0; x < 20; ++x) {
            d_info[14 + x] = float(ind[x]);
            d_info[14 + x + 20] = float(dis[x]);
        }
        
    }
#endif
    // sum and average
    for(int k = 0; k < x_height; ++k) {
        X_d[k* x_width + off] = 0;
    }
    for (int p = 0; p < num; ++p) {
        for(int k = 0; k < x_height; ++k)
            X_d[k * x_width + off] +=  X[k * x_width + resInd[p]];
    }
    for(int k = 0; k < x_height; ++k)
        X_d[k * x_width + off] /= num;
}

/*
 * Host code
 */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    /* Declare all variables.*/
    mxGPUArray const *X;
    mxGPUArray * Xd;
    
    double const * d_X;
    double * d_Xd;
    double * ptr;
    
    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";
    
    /* Choose a reasonably sized number of threads for the block. */
    // Don't set threadsPerBlock too big or the shared memory may exceed block
    // shared memory limit and cause CUDA_ILLEGAL_ADDRESS error !!!
    int const threadsPerBlock = 32;
    int blocksPerGrid = 1024;
    
    /* Initialize the MathWorks GPU API. */
    mxInitGPU();
    /* Throw an error if the input is not a GPU array. */
    if ((nrhs != 8) || !(mxIsGPUArray(prhs[0]))) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }
    
    X = mxGPUCreateFromMxArray(prhs[0]);
    ptr = mxGetPr(prhs[1]);
    int x_height = int(ptr[0] + 0.5);
    ptr = mxGetPr(prhs[2]);
    int N = int(ptr[0] + 0.5);
    ptr = mxGetPr(prhs[3]);
    int M = int(ptr[0] + 0.5);
    ptr = mxGetPr(prhs[4]);
    int b2 = int(ptr[0] + 0.5);
    ptr = mxGetPr(prhs[5]);
    int wnd = int(ptr[0] + 0.5);
    ptr = mxGetPr(prhs[6]);
    int step = int(ptr[0] + 0.5);
    ptr = mxGetPr(prhs[7]);
    int num = int(ptr[0] + 0.5);
    
    int  X_length = (int)(mxGPUGetNumberOfElements(X));
    int x_width = X_length / x_height;
    
#ifdef DEBUG
    printf("nrhs = %d, x_height = %d, N = %d, M = %d, b2 = %d, wnd = %d, step = %d, num = %d\n", nrhs, x_height, N, M, b2, wnd, step, num);
#endif
    
#ifdef DEBUG
    printf("mxGPUGetClassID(A) = %d, mxDOUBLE_CLASS = %d\n", mxGPUGetClassID(X), mxDOUBLE_CLASS);
#endif
    
    // Verify that X really is a double array before extracting the pointer.
    if (mxGPUGetClassID(X) != mxDOUBLE_CLASS) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }
    
#ifdef DEBUG
    printf("break point 1\n");
#endif
    
    /*  Extract a pointer to the input data on the device. */
    d_X = (double const *)(mxGPUGetDataReadOnly(X));
    
    /* Create a GPUArray to hold the result and get its underlying pointer. */
    Xd = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(X),
                             mxGPUGetDimensions(X),
                             mxGPUGetClassID(X),
                             mxGPUGetComplexity(X),
                             MX_GPU_DO_NOT_INITIALIZE);
    d_Xd = (double *)(mxGPUGetData(Xd));
    
#ifdef DEBUG
    printf("break point 2\n");
#endif
    
    float* d_info = NULL;
    float info[100];
    cudaMalloc((void **)&d_info, 100 *sizeof(float));
    cudaMemcpy(d_info, info, 100 *sizeof(float), cudaMemcpyHostToDevice);
#ifdef DEBUG
    printf("bpg = %d, tpb = %d\n", blocksPerGrid, threadsPerBlock);
#endif
    
    int thread_num = blocksPerGrid * threadsPerBlock;
    for(int startThreadNum = 0; startThreadNum < N*M; startThreadNum += thread_num) {
        denoiseKernel<<<blocksPerGrid, threadsPerBlock>>>(startThreadNum, d_X, x_height, x_width, N, M, b2, wnd, step, num,  d_Xd, d_info);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }
#ifdef DEBUG
    printf("break point 3\n");
    cudaMemcpy(info, d_info, 100 *sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < 55; ++i)
        printf("info[%d] = %f\n", i, info[i]);
#endif
    cudaFree(d_info);
    /* Wrap the result up as a MATLAB gpuArray for return. */
    plhs[0] = mxGPUCreateMxArrayOnGPU(Xd);
    /*
     * The mxGPUArray pointers are host-side structures that refer to device
     * data. These must be destroyed before leaving the MEX function.
     */
    mxGPUDestroyGPUArray(X);
    mxGPUDestroyGPUArray(Xd);
}
