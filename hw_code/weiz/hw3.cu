#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <assert.h>
#define BASE_THREAD_NUM 256

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);
#define cudacall(call)                                                         \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (cudaSuccess != err) {                                                  \
      fprintf(stderr, "CUDA Error:\nFile = %s\nLine = %d\nReason = %s\n",      \
              __FILE__, __LINE__, cudaGetErrorString(err));                    \
      cudaDeviceReset();                                                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define cublascall(call)                                                       \
  do {                                                                         \
    cublasStatus_t err = (call);                                               \
    if (CUBLAS_STATUS_SUCCESS != err) {                                        \
      fprintf(stderr, "CUBLAS Error:\nFile = %s\nLine = %d\nReason = %d\n",    \
              __FILE__, __LINE__, err);                                        \
      cudaDeviceReset();                                                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)
struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }

  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

void copyToHost(scalar_t *host_ptr, scalar_t *device_ptr, size_t n){
  cudacall(cudaMemcpy(host_ptr, device_ptr, n*sizeof(scalar_t), cudaMemcpyDeviceToHost));
}

__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[gid] = a[gid] + b[gid];
    //printf("out[%d]: %f ", gid, out[gid]);
  }
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}


__device__ scalar_t _mul(scalar_t a, scalar_t b){
	return a*b;
}

typedef scalar_t (*binary_func) (scalar_t, scalar_t);
__device__ binary_func bfunc[1]={_mul};

//template <typename F>
__global__ void EwiseFuncKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size, int ftype){
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  //printf("gid %d ", gid);
  if (gid < size) {
    out[gid] = bfunc[ftype](a[gid], b[gid]);
  }
}

//template <typename F>
void EwiseFunc(const CudaArray& a, const CudaArray& b, CudaArray* out, int ftype){
  assert(a.size == b.size);
	assert(a.size == out->size);
  CudaDims dim = CudaOneDim(out->size);
  EwiseFuncKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size, ftype);
}

/* 
template < typename T,0> twoOperandsFuncWrapper(T input1, T input2) {
  EwiseFunc<T>(intput1, _mul)
}
*/
void EwiseMul(const CudaArray& a, const CudaArray& b, CudaArray* out) {
	EwiseFunc(a, b, out, 0);
}
/**
 * Test EwiseAdd
*/
void test1(){
  size_t sz = 100;
  CudaArray a(sz);
  Fill(&a, 1);
  CudaArray b(sz);
  Fill(&b,2);
  CudaArray c(sz);
  Fill(&c,0);
  EwiseAdd(a, b, &c);
  scalar_t * host_ptr = (scalar_t *) malloc(sizeof(scalar_t)*sz);
  copyToHost(host_ptr, c.ptr, sz);
  for(size_t i = 0; i < sz; ++i){
    std::cout<<host_ptr[i]<<" ";
  }
  std::cout<<std::endl;
}

/**
 * Test EwiseMul
*/
void test2(){
  size_t sz = 10;
  CudaArray a(sz);
  Fill(&a, 1);
  CudaArray b(sz);
  Fill(&b,2);
  CudaArray c(sz);
  Fill(&c,0);
  EwiseMul(a, b, &c);
  scalar_t * host_ptr = (scalar_t *) malloc(sizeof(scalar_t)*sz);
  copyToHost(host_ptr, c.ptr, sz);
  for(size_t i = 0; i < sz; ++i){
    std::cout<<host_ptr[i]<<" ";
  }
  std::cout<<std::endl;
}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  /// BEGIN SOLUTION
  assert(false && "Not Implemented");
  /// END SOLUTION
}

int main(int argc, char **argv){
  //test1();
  test2();
  return 0;
}
