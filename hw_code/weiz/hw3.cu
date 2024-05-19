#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <assert.h>
#include <cmath>
#define BASE_THREAD_NUM 256
#define BASE_THREAD_NUM_2D 16 
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

CudaDims CudaTwoDim(size_t X, size_t Y){
  CudaDims dim;
  size_t num_blocks_along_x = ceil((float) X / (float) BASE_THREAD_NUM_2D);
  size_t num_blocks_along_y = ceil((float) Y / (float) BASE_THREAD_NUM_2D);
  dim.block = dim3(BASE_THREAD_NUM_2D, BASE_THREAD_NUM_2D, 1);
  dim.grid = dim3(num_blocks_along_x, num_blocks_along_y, 1);
  std::cout<<"num_blocks_along_x: "<<num_blocks_along_x<<" num_blocks_along_y: "<<num_blocks_along_y<<std::endl;
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


enum BinaryOp{
  MUL,
  DIV,
  POW,
  MAX,
  EQ,
  GE
};

__device__ scalar_t _mul(scalar_t a, scalar_t b){
	return a*b;
}
__device__ scalar_t _div(scalar_t a, scalar_t b){
	return a/b;
}
__device__ scalar_t _power(scalar_t a, scalar_t b){
	return std::pow(a,b);
}
__device__ scalar_t _max(scalar_t a, scalar_t b){
  return max(a,b); // it is a pity that std::pow() in c++11 is not constexpr
}
__device__ scalar_t _eq(scalar_t a, scalar_t b){
	return a==b;
}
__device__ scalar_t _ge(scalar_t a, scalar_t b){
	return a>=b;
}
typedef scalar_t (*binary_func) (scalar_t, scalar_t);
__device__ binary_func bfunc[6]={_mul, _div,_power, _max, _eq, _ge};

enum UnaryOp{
  LOG,
  EXP,
  TANH
};

__device__ scalar_t _log(scalar_t a){
  return std::log(a);
}
__device__ scalar_t _exp(scalar_t a){
	return std::exp(a);
}
__device__ scalar_t _tanh(scalar_t a){
	return std::tanh(a);
}
typedef scalar_t (*unary_func) (scalar_t);
__device__ unary_func ufunc[3]={_log, _exp, _tanh};

//template <typename F>
__global__ void EwiseFuncKernel(const scalar_t* a, const scalar_t* b, const scalar_t b_val, scalar_t* out, size_t size, int ftype){
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  //printf("gid %d ", gid);
  if (gid < size) {
    if(b!=NULL){
      out[gid] = bfunc[ftype](a[gid], b[gid]);
    }else{
      out[gid] = bfunc[ftype](a[gid], b_val);
    }
  }
}


//template <typename F>
void EwiseFunc(const scalar_t *a_ptr, const scalar_t *b_ptr, const scalar_t b_val, CudaArray* out, BinaryOp ftype){
  CudaDims dim = CudaOneDim(out->size);
  EwiseFuncKernel<<<dim.grid, dim.block>>>(a_ptr, b_ptr, b_val, out->ptr, out->size, ftype);
}



__global__ void SingleEwiseFuncKernel(const scalar_t* a, scalar_t* out, size_t size, UnaryOp ftype){
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  //printf("gid %d ", gid);
  if (gid < size) {
    out[gid] = ufunc[ftype](a[gid]);
  }
}

// for log exp and tanh
void SingleEwiseFunc(scalar_t* a_ptr, CudaArray* out, UnaryOp ftype){
  CudaDims dim = CudaOneDim(out->size);
  SingleEwiseFuncKernel<<<dim.grid, dim.block>>>(a_ptr,out->ptr, out->size, ftype);
}

/* 
template < typename T,0> twoOperandsFuncWrapper(T input1, T input2) {
  EwiseFunc<T>(intput1, _mul)
}
*/

void EwiseMul(const CudaArray& a, const CudaArray& b, CudaArray* out) {
	EwiseFunc(a.ptr, b.ptr, 0, out, MUL);
}
void ScalarMul(const CudaArray& a, scalar_t val, CudaArray* out) {
	EwiseFunc(a.ptr, NULL, val, out, MUL);
}


void EwiseDiv(const CudaArray& a, const CudaArray& b, CudaArray* out) {
	EwiseFunc(a.ptr, b.ptr, 0, out, DIV);
}

void ScalarDiv(const CudaArray& a, scalar_t val, CudaArray* out) {
	EwiseFunc(a.ptr, NULL, val, out, DIV);
}


void ScalarPower(const CudaArray& a, scalar_t val, CudaArray* out) {
	EwiseFunc(a.ptr, NULL, val, out, POW);
}

void EwiseMaximum(const CudaArray& a, const CudaArray& b, CudaArray* out) {
	EwiseFunc(a.ptr, b.ptr, 0, out, MAX);
}

void ScalarMaximum(const CudaArray& a, scalar_t val, CudaArray* out) {
	EwiseFunc(a.ptr, NULL, val,out, MAX);
}


void EwiseEq(const CudaArray& a, const CudaArray& b, CudaArray* out) {
	EwiseFunc(a.ptr, b.ptr, 0, out, EQ);
}

void ScalarEq(const CudaArray& a, scalar_t val, CudaArray* out) {
	EwiseFunc(a.ptr, NULL, val, out, EQ);
}


void EwiseGe(const CudaArray& a, const CudaArray& b, CudaArray* out) {
	EwiseFunc(a.ptr, b.ptr,0, out, GE);
}

void ScalarGe(const CudaArray& a, scalar_t val, CudaArray* out) {
	EwiseFunc(a.ptr, NULL, val, out, GE);
}


void EwiseLog(const CudaArray& a,CudaArray* out){
	SingleEwiseFunc(a.ptr, out, LOG);
}



void EwiseExp(const CudaArray& a,CudaArray* out){
	SingleEwiseFunc(a.ptr, out, EXP);
}



void EwiseTanh(const CudaArray& a,CudaArray* out){
	SingleEwiseFunc(a.ptr, out, TANH);
}

/**
 * output is M x P
*/
__global__ void MatMulKernel(const scalar_t* a_ptr, const scalar_t* b_ptr, scalar_t* out, 
uint32_t M, uint32_t N, uint32_t P){
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x  + threadIdx.x;
  if (row < M && col < P){
    size_t flatten_idx = row * P + col;
    out[flatten_idx] = 0;
    for(size_t k = 0; k < N; ++k){
      scalar_t tmp = a_ptr[row*N+k]* b_ptr[k*P+col];
      out[flatten_idx]+= tmp;
    }
  }

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
  CudaDims dim = CudaTwoDim(P, M);
  MatMulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
  //assert(false && "Not Implemented");
  /// END SOLUTION
}



scalar_t _sum(scalar_t a, scalar_t b){
  return a+b;
}
enum ReduceOP{
  REDUCE_MAX, 
  REDUCE_SUM
};

__device__ binary_func reduce_func[2]={_max,_sum};

__global__ void ReduceTemplateFuncKernel(const scalar_t *view, scalar_t *out, size_t out_size, size_t reduce_size, ReduceOP reduce_op, scalar_t reduce_id){
  // TODO!!!
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if(gid < out_size){
    scalar_t _reduce_result = reduce_id;
    for(size_t i=0; i < reduce_size; ++i){
      size_t offset = gid*reduce_size+i;
      _reduce_result = reduce_func[reduce_op](_reduce_result, view[offset]); // note offset will never go over bound, as we did gid check already
    }
    out[gid] = _reduce_result;
  }
}
  



void ReduceTemplateFunc(const CudaArray& a, CudaArray* out, size_t reduce_size, ReduceOP reduce_op, scalar_t reduce_id){
	assert(a.size == out->size * reduce_size);
  CudaDims dim = CudaOneDim(out->size);
  ReduceTemplateFuncKernerl<<<dim.grid, dim.block>>>(a.ptr,out->ptr, out->size, reduce_size, reduce_op, reduce_id);
	
}










void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  ReduceTemplateFunc(a, out, reduce_size, REDUCE_MAX, std::numeric_limits<scalar_t>::lowest());
  /// END SOLUTION
}



void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  ReduceTemplateFunc(a, out, reduce_size, REDUCE_SUM, 0);
  /// END SOLUTION
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

/**
 * Test ScalMul
*/
void test3(){
  size_t sz = 10;
  CudaArray a(sz);
  Fill(&a, 1);
  
  CudaArray c(sz);
  Fill(&c,0);
  ScalarMul(a, 2.5, &c);
  scalar_t * host_ptr = (scalar_t *) malloc(sizeof(scalar_t)*sz);
  copyToHost(host_ptr, c.ptr, sz);
  for(size_t i = 0; i < sz; ++i){
    std::cout<<host_ptr[i]<<" ";
  }
  std::cout<<std::endl;
}

/**
 * Test Tanh
*/
void test4(){
  size_t sz = 10;
  CudaArray a(sz);
  Fill(&a, 1);
  
  CudaArray c(sz);
  Fill(&c,0);
  EwiseTanh(a, &c);
  scalar_t * host_ptr = (scalar_t *) malloc(sizeof(scalar_t)*sz);
  copyToHost(host_ptr, c.ptr, sz);
  for(size_t i = 0; i < sz; ++i){
    std::cout<<host_ptr[i]<<" ";
  }
  std::cout<<std::endl;
}

void test5(){
  size_t sz = 64;
  CudaArray a(sz);
  Fill(&a, 1);
  CudaArray b(sz);
  Fill(&b, 1);
  CudaArray c(sz);
  Fill(&c,0);
  Matmul(a,b,&c, 8,8,8);
  scalar_t * host_ptr = (scalar_t *) malloc(sizeof(scalar_t)*sz);
  copyToHost(host_ptr, c.ptr, sz);
  size_t idx=0;
  for(size_t i = 0; i < 8; ++i){
    for(size_t j=0; j < 8; ++j){
      std::cout<<host_ptr[idx++]<<" ";
    }
    std::cout<<std::endl;
  }
  std::cout<<std::endl;

}

void test6(){
  size_t sz = 64*64;
  CudaArray a(sz);
  Fill(&a, 1);
  CudaArray b(sz);
  Fill(&b, 1);
  CudaArray c(sz);
  Fill(&c,0);
  Matmul(a,b,&c, 64,64,64);
  scalar_t * host_ptr = (scalar_t *) malloc(sizeof(scalar_t)*sz);
  copyToHost(host_ptr, c.ptr, sz);
  size_t idx=0;
  for(size_t i = 0; i < 64; ++i){
    for(size_t j=0; j < 64; ++j){
      std::cout<<host_ptr[idx++]<<" ";
    }
    std::cout<<std::endl;
  }
 
}

/**
 * Test reduce sum
*/
void test7(){
  size_t sz = 64;
  CudaArray view(sz);
  Fill(&view, 1);
  size_t out_size = 8;
  CudaArray out(out_size);
  ReduceSum(view, *out, sz/out_size);
  scalar_t * host_ptr = (scalar_t *) malloc(sizeof(scalar_t)*out_size);
  copyToHost(host_ptr, out.ptr, out_size);
  size_t idx=0;
  for(size_t i = 0; i < out_size; ++i){
    std::cout<<host_ptr[i]<<" ";
  }
  std::cout<<std::endl;
}

int main(int argc, char **argv){
  //test1();
  //test2();
  //test3();
  //test4();
  //test5();
  //test6();
  test7();
  return 0;
}
