#include <cmath>
#include <iostream>
#include <stdexcept>
#include <assert.h>
#include <vector>
#include <limits>
#include <string.h>
/*#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stddef.h>*/
#define ALIGNMENT 256
#define TILE 2
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);
/**
 * This is a utility structure for maintaining an array aligned to ALIGNMENT boundaries in
 * memory.  This alignment should be at least TILE * ELEM_SIZE, though we make it even larger
 * here by default.
 */
struct AlignedArray {
  AlignedArray(const size_t size) {
    int ret = posix_memalign((void**)&ptr, ALIGNMENT, size * ELEM_SIZE);
    if (ret != 0) throw std::bad_alloc();
    this->size = size;
  }
  ~AlignedArray() { free(ptr); }
  size_t ptr_as_int() {return (size_t)ptr; }
  scalar_t* ptr;
  size_t size;
};



/**
 * remove the top, decrement and reload
 */
void pop_dec_reload(std::vector<int32_t> & repr){
	assert(!repr.empty());
	int32_t top = repr.back();
	repr.pop_back();
	top--;
	assert(top>=-1); // it could be -1 which means the highest dimension has been explored entirely
	repr.push_back(top);
}

/**
 * repr is not a full vector
 * remove the top, dec, , need to repopulate the higher dimensions
 */
void pop_dec_repopulate(std::vector<int32_t> & repr,  std::vector<int32_t> & shape){
	assert(repr.size() < shape.size());
	int32_t top = repr.back();
	assert(top >= 0);
	repr.pop_back();
	top--;
	if(top >= 0){
		repr.push_back(top);
		size_t start_dim = repr.size();
		for(size_t i = start_dim; i < shape.size(); ++i){
			int32_t s = shape.at(i) -1;
			repr.push_back(s-1);
		}
		assert(repr.size() == shape.size()); // repr is populated
	}else{
		repr.push_back(top); // last lower dimension has also been exhausted
	}
	return;
}


/**
 * return strides for compact tensor, given tensor's shape
 */
std::vector<int32_t> get_compact_strides(std::vector<int32_t> shape){
	size_t dim = shape.size();
	std::vector<int> compact_strides(dim, 0);
	compact_strides[dim-1] = 1; // dst_strides highest dimension is always 1, as it is always compact
	size_t stride_at_dim = 1;
	for(int i = dim-2; i >=0 ; --i){ // get strides for output tensor (always compact), had two bugs here:(1) use int i to avoid
		// underflow, (ii) use my own formula to calculate strides
		stride_at_dim *= shape[i+1];
		compact_strides[i] = stride_at_dim;
	}
	return compact_strides;
}

void fill_tensor_at(scalar_t * dst_ptr, const scalar_t * src_ptr,
		std::vector<int32_t> & dst_strides, std::vector<int32_t> & src_strides,
		std::vector<int32_t> & repr, scalar_t src_val=0){
	assert(repr.size() == src_strides.size());
	assert(repr.size() == dst_strides.size());
	size_t src_ptr_offset = 0;
	size_t dst_ptr_offset = 0;
	for (size_t i = 0; i < repr.size(); ++i){
		src_ptr_offset += repr[i]*src_strides[i];
		dst_ptr_offset += repr[i]*dst_strides[i];
	}
	if(src_ptr != NULL){
		*(dst_ptr+dst_ptr_offset) = *(src_ptr+src_ptr_offset);
	}else{
		*(dst_ptr+dst_ptr_offset) = src_val;
	}
	//std::cout<<"!!! "<<"src offset: "<<src_ptr_offset<<" val: "<<*(src_ptr+src_ptr_offset)<<std::endl;
	//std::cout<<"!!! "<<"dst offset: "<<dst_ptr_offset<<" val: "<<*(dst_ptr+dst_ptr_offset)<<std::endl;
}

/**
 * called when the last dimension is exhausted,
 * return the dimension where it will start exploring, -1 if the entire repr space is explored.
 */
int reload(std::vector<int32_t> & repr, std::vector<int32_t> & shape){
	size_t dim = repr.size();
	assert(dim == shape.size());
	assert(repr[dim-1]==-1);
	if(dim==1){ // repr is just one element
		return -1;
	}
	// find the highest dimension i where repr[i] is not zero, decrement and start exploring
	int highest_dim_not_zero_idx = dim-2;
	while( (highest_dim_not_zero_idx>=0) && repr[highest_dim_not_zero_idx]==0){
		highest_dim_not_zero_idx--;
	}
	if(highest_dim_not_zero_idx>=0){ // has more space to explore
		repr[highest_dim_not_zero_idx]--;
		for(size_t i = highest_dim_not_zero_idx+1; i < dim; ++i){
			repr[i] = shape[i]-1;
		}
		return highest_dim_not_zero_idx;
	}else{ // no more space to explore
		assert(highest_dim_not_zero_idx==-1);
		return highest_dim_not_zero_idx;
	}
}

void tensor_assign(scalar_t *dest_ptr, const scalar_t *src_ptr,size_t dest_offset, size_t src_offset,
		std::vector<int32_t> dest_strides, std::vector<int32_t> src_strides,
		std::vector<int32_t> shape,
		scalar_t src_val = 0
		){

	// step1 initialize num_of_elements and repr
	size_t num_of_elements = 1;
	std::vector<int32_t> repr;
	for(std::vector<int32_t>::iterator it = shape.begin(); it != shape.end(); ++it){
		num_of_elements *= (*it);
		repr.push_back(*it-1);
	}
	size_t dim = shape.size();
	scalar_t * real_dest_ptr = dest_ptr + dest_offset;
	const scalar_t * real_src_ptr = src_ptr + src_offset;

	// step 2 iterate repr
	size_t num_elem_processed = 0;
	while(num_elem_processed < num_of_elements){
		//fill_tensor_at(repr, strides, dst_strides, src_ptr, dst_ptr);
		fill_tensor_at(real_dest_ptr, real_src_ptr, dest_strides, src_strides, repr, src_val);
		num_elem_processed++;
		repr[dim-1]--;
		if(repr[dim-1] == -1){ // when the highest dimension has been explored
			int res = reload(repr, shape);
			if (res == -1 ){
				assert(num_elem_processed == num_of_elements);
				return;
			}
		}
	}
	return;
}

void Compact(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory
   *
   * Args:
   *   a: non-compact representation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   *
   * Returns:
   *  void (you need to modify out directly, rather than returning anything; this is true for all the
   *  function will implement here, so we won't repeat this note.)
   */
  /// BEGIN SOLUTION
	//std::cout<<"step 0"<<std::endl;
	// step 1 set up
	//scalar_t *src_ptr = a.ptr+offset;
	//scalar_t *dst_ptr = out->ptr;
	//assert(shape.size() == strides.size());
	//size_t dim = shape.size();
	std::vector<int32_t> dst_strides = get_compact_strides(shape);
	tensor_assign(out->ptr, a.ptr, 0, offset, dst_strides, strides, shape);
	return;
}

void EwiseSetitem(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array
   *
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
	std::vector<int32_t> dst_strides = strides;
	std::vector<int32_t> src_strides = get_compact_strides(shape);
	scalar_t * dst_ptr = out->ptr; // offset is dst ptr offset
	const scalar_t * src_ptr = a.ptr;
	tensor_assign(dst_ptr, src_ptr, offset, 0, dst_strides, src_strides, shape);
  /// END SOLUTION
}

void ScalarSetitem(const size_t size, scalar_t val, AlignedArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   *
   * Args:
   *   size: number of elements to write in out array (note that this will not be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
   *         product of items in shape, but convenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */

  /// BEGIN SOLUTION
	std::vector<int32_t> dst_strides = strides;
	std::vector<int32_t> src_strides = get_compact_strides(shape);
	scalar_t * dst_ptr = out->ptr; // offset is dst ptr offset
	tensor_assign(out->ptr, NULL, offset, 0, dst_strides, src_strides, shape, val);
  /// END SOLUTION
}

void test1(){
	// step1 prepare a
	size_t sz = 6;
	AlignedArray a(sz);
	for(size_t i = 0; i < sz; ++i){
		a.ptr[i] = (scalar_t) i;
	}
	std::vector<int32_t> strides = {3,1}; // now a is 2x3: [0,1,2]
	                                      //               [3,4,5]
	std::cout<<strides.size()<<std::endl;
	// step2 prepare b
	AlignedArray b(2);
	std::vector<int32_t> shape = {2,1}; //  b is [0,
	                                          //  3]
	std::cout<<shape.size()<<std::endl;
	std::cout<<"shape"<<std::endl;
	for(std::vector<int32_t>::iterator it = shape.begin(); it != shape.end(); ++it){
		std::cout<<*it<<std::endl;
	}
	std::cout<<"end of shape"<<std::endl;
	Compact(a, &b, shape, strides, 0);
	size_t idx=0;
	std::cout<<"Result"<<std::endl;
	/*for(std::vector<int32_t>::iterator it = shape.begin(); it != shape.end(); ++it){
		for(size_t j = 0; j < *it; ++j){
			std::cout<<b.ptr[idx++]<<" ";
		}
		std::cout<<std::endl;
	}*/
	for (size_t i = 0; i < 2; ++i){
		std::cout<<b.ptr[idx++]<<std::endl;
	}

}

void test2(){
	// step1 prepare a
	size_t sz = 9;
	AlignedArray a(sz);
	for(size_t i = 0; i < sz; ++i){
		a.ptr[i] = (scalar_t) i;
	}
	std::vector<int32_t> strides = {3,1}; // now a is 3x3
	std::cout<<strides.size()<<std::endl;
	// step2 prepare b
	AlignedArray b(6);
	std::vector<int32_t> shape = {3,2};
	std::cout<<shape.size()<<std::endl;
	std::cout<<"shape"<<std::endl;
	for(std::vector<int32_t>::iterator it = shape.begin(); it != shape.end(); ++it){
		std::cout<<*it<<std::endl;
	}
	std::cout<<"end of shape"<<std::endl;
	Compact(a, &b, shape, strides, 0);
	size_t idx=0;
	std::cout<<"Result"<<std::endl;
	/*for(std::vector<int32_t>::iterator it = shape.begin(); it != shape.end(); ++it){
		for(size_t j = 0; j < *it; ++j){
			std::cout<<b.ptr[idx++]<<" ";
		}
		std::cout<<std::endl;
	}*/
	for (size_t i = 0; i < 6; ++i){
		std::cout<<b.ptr[idx++]<<std::endl;
	}

}

void test3(){
	// step1 prepare a
	size_t sz = 6;
	AlignedArray a(sz);
	for(size_t i = 0; i < sz; ++i){
		a.ptr[i] = (scalar_t) -1;
	}
	std::vector<int32_t> strides = {3,1}; // now a is 2x3
	//std::cout<<strides.size()<<std::endl;
	// step2 prepare b
	AlignedArray b(2);
	std::vector<int32_t> shape = {2,1}; // now
	for(size_t i = 0; i < 2; ++i){
		b.ptr[i] = i;
	}
	EwiseSetitem(b, &a, shape, strides, 0);
	size_t idx=0;
	std::cout<<"Result"<<std::endl;
	/*for(std::vector<int32_t>::iterator it = shape.begin(); it != shape.end(); ++it){
		for(size_t j = 0; j < *it; ++j){
			std::cout<<b.ptr[idx++]<<" ";
		}
		std::cout<<std::endl;
	}*/
	for (size_t i = 0; i < 6; ++i){
		std::cout<<a.ptr[idx++]<<std::endl;
	}

}

void test4(){
	// step1 prepare a
	size_t sz = 9;
	AlignedArray a(sz);
	for(size_t i = 0; i < sz; ++i){
		a.ptr[i] = (scalar_t) -1;
	}
	std::vector<int32_t> strides = {3,1}; // now a is 3x3
	//std::cout<<strides.size()<<std::endl;
	// step2 prepare b
	AlignedArray b(6);
	std::vector<int32_t> shape = {3,2}; // now
	for(size_t i = 0; i < 6; ++i){
		b.ptr[i] = i;
	}
	EwiseSetitem(b, &a, shape, strides, 0);
	size_t idx=0;
	std::cout<<"Result"<<std::endl;
	for (size_t i = 0; i < 9; ++i){
		std::cout<<a.ptr[idx++]<<std::endl;
	}
}

void test5(){
	// step1 prepare a
	size_t sz = 6;
	AlignedArray a(sz);
	for(size_t i = 0; i < sz; ++i){
		a.ptr[i] = (scalar_t) -1;
	}
	std::vector<int32_t> strides = {3,1}; // now a is 2x3
	//std::cout<<strides.size()<<std::endl;
	// step2 prepare b
	AlignedArray b(2);
	std::vector<int32_t> shape = {2,1}; // now
	for(size_t i = 0; i < 2; ++i){
		b.ptr[i] = i;
	}
	ScalarSetitem(2, 100, &a, shape, strides, 0);
	size_t idx=0;
	std::cout<<"Result"<<std::endl;
	/*for(std::vector<int32_t>::iterator it = shape.begin(); it != shape.end(); ++it){
		for(size_t j = 0; j < *it; ++j){
			std::cout<<b.ptr[idx++]<<" ";
		}
		std::cout<<std::endl;
	}*/
	for (size_t i = 0; i < 6; ++i){
		std::cout<<a.ptr[idx++]<<std::endl;
	}

}

void test6(){
	// step1 prepare a
	size_t sz = 9;
	AlignedArray a(sz);
	for(size_t i = 0; i < sz; ++i){
		a.ptr[i] = (scalar_t) -1;
	}
	std::vector<int32_t> strides = {3,1}; // now a is 3x3
	//std::cout<<strides.size()<<std::endl;
	// step2 prepare b
	AlignedArray b(6);
	std::vector<int32_t> shape = {3,2}; // now
	for(size_t i = 0; i < 6; ++i){
		b.ptr[i] = i;
	}
	ScalarSetitem(6, 100, &a, shape, strides, 0);
	size_t idx=0;
	std::cout<<"Result"<<std::endl;
	for (size_t i = 0; i < 9; ++i){
		std::cout<<a.ptr[idx++]<<std::endl;
	}
}

void EwiseAdd(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + b.ptr[i];
  }
}

void ScalarAdd(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + val;
  }
}

template <typename F>
void EwiseFunc(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, F f){
	assert(a.size == b.size);
	assert(a.size == out->size);
	for (size_t i = 0; i < a.size; i++) {
		out->ptr[i] = f(a.ptr[i], b.ptr[i]);
	}
}

template <typename F>
void ScalarFunc(const AlignedArray& a, scalar_t val, AlignedArray* out, F f){
	assert(a.size == out->size);
	for (size_t i = 0; i < a.size; i++) {
		out->ptr[i] = f(a.ptr[i], val);
	}
}

// for log exp and tanh
template <typename F>
void SingleEwiseFunc(const AlignedArray& a, AlignedArray* out, F f){
	assert(a.size == out->size);
	for(size_t i = 0; i < a.size; ++i){
		out->ptr[i] = f(a.ptr[i]);
	}
}

/**
 * In the code the follows, use the above template to create analogous element-wise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */

scalar_t _mul(scalar_t a, scalar_t b){
	return a*b;
}
void EwiseMul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
	EwiseFunc(a, b, out, _mul);
}

void ScalarMul(const AlignedArray& a, scalar_t val, AlignedArray* out) {
	ScalarFunc(a, val, out, _mul);
}
scalar_t _div(scalar_t a, scalar_t b){
	return a/b;
}
void EwiseDiv(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
	EwiseFunc(a, b, out, _div);
}

void ScalarDiv(const AlignedArray& a, scalar_t val, AlignedArray* out) {
	ScalarFunc(a, val, out, _div);
}
scalar_t _power(scalar_t a, scalar_t b){
	return std::pow(a,b);
}

void ScalarPower(const AlignedArray& a, scalar_t val, AlignedArray* out) {
	ScalarFunc(a, val, out, _power);
}
scalar_t _max(scalar_t a, scalar_t b){
	return std::max(a,b);
}
void EwiseMaximum(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
	EwiseFunc(a, b, out, _max);
}

void ScalarMaximum(const AlignedArray& a, scalar_t val, AlignedArray* out) {
	ScalarFunc(a,val,out, _max);
}

scalar_t _eq(scalar_t a, scalar_t b){
	return a==b;
}
void EwiseEq(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
	EwiseFunc(a, b, out, _eq);
}

void ScalarEq(const AlignedArray& a, scalar_t val, AlignedArray* out) {
	ScalarFunc(a,val,out, _eq);
}

scalar_t _ge(scalar_t a, scalar_t b){
	return a>=b;
}
void EwiseGe(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
	EwiseFunc(a, b, out, _ge);
}

void ScalarGe(const AlignedArray& a, scalar_t val, AlignedArray* out) {
	ScalarFunc(a,val,out, _ge);
}

scalar_t _log(scalar_t a){
	return std::log(a);
}

void EwiseLog(const AlignedArray& a,AlignedArray* out){
	SingleEwiseFunc(a, out, _log);
}

scalar_t _exp(scalar_t a){
	return std::exp(a);
}

void EwiseExp(const AlignedArray& a,AlignedArray* out){
	SingleEwiseFunc(a, out, _exp);
}

scalar_t _tanh(scalar_t a){
	return std::tanh(a);
}

void EwiseTanh(const AlignedArray& a,AlignedArray* out){
	SingleEwiseFunc(a, out, _tanh);
}

template<typename F, typename I>
void ReduceTemplateFunc(const AlignedArray& a, AlignedArray* out, size_t reduce_size, F reduce_func, I reduce_id){
	assert(a.size == out->size * reduce_size);
	size_t out_ptr_idx=0;
	for(size_t i = 0; i < a.size; i+=reduce_size){
		scalar_t _reduce_result = reduce_id;
		for(size_t j=i; j < i+reduce_size;++j){
			_reduce_result = reduce_func(_reduce_result, a.ptr[j]);
		}
		out->ptr[out_ptr_idx++] = _reduce_result;
	}
}



void ReduceMax(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN SOLUTION
	ReduceTemplateFunc(a, out, reduce_size, _max, std::numeric_limits<scalar_t>::lowest());
  /// END SOLUTION

}


scalar_t _sum(scalar_t a, scalar_t b){
	return a+b;
}

void ReduceSum(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking sum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN SOLUTION
	ReduceTemplateFunc(a, out, reduce_size, _sum, 0);
  /// END SOLUTION
}

void Matmul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m, uint32_t n,
            uint32_t p) {
  /**
   * Multiply two (compact) matrices into an output (also compact) matrix.  For this implementation
   * you can use the "naive" three-loop algorithm.
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: compact 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   */

  /// BEGIN SOLUTION
  assert(a.size == m*n);
  assert(b.size == n*p);
  assert(out->size == m*p);
  for(size_t i = 0; i < m; ++i){
	  for(size_t j = 0; j < p; ++j){
		  out->ptr[i*p+j] = 0;
		  for(size_t k = 0; k < n; ++k){
			  out->ptr[i*p+j] += a.ptr[i*n+k] * b.ptr[k*p+j];
		  }
	  }
  }
  /// END SOLUTION
}


void test7(){
	size_t sz = 6;
	AlignedArray a(sz);
	for(size_t i = 0; i < sz; ++i){
		a.ptr[i] = (scalar_t) i;
	}
	AlignedArray b(sz);
	for(size_t i = 0; i < sz; ++i){
		b.ptr[i] = (scalar_t) i;
	}
	AlignedArray c(sz);
	EwiseMul(a, b, &c);
	for(size_t i = 0; i < sz; ++i){
		std::cout<<c.ptr[i]<<" ";
	}
	std::cout<<std::endl;
}

void test8(){
	size_t sz = 6;
	AlignedArray a(sz);
	for(size_t i = 0; i < sz; ++i){
		a.ptr[i] = (scalar_t) i;
	}
	AlignedArray c(sz);
	ScalarMul(a, 10, &c);
	for(size_t i = 0; i < sz; ++i){
		std::cout<<c.ptr[i]<<" ";
	}
	std::cout<<std::endl;
}

void test9(){
	size_t sz = 6;
	AlignedArray a(sz);
	for(size_t i = 0; i < sz; ++i){
		a.ptr[i] = (scalar_t) i;
	}
	AlignedArray out(2);
	size_t reduce_size = 3;
	std::cout<<"before reduce_max"<<std::endl;
	ReduceMax(a, &out, reduce_size);
	for(size_t i = 0; i < 2; ++i){
		std::cout<<out.ptr[i]<<" ";
	}
	std::cout<<std::endl;

}

void test10(){
	size_t sz = 6;
	AlignedArray a(sz);
	for(size_t i = 0; i < sz; ++i){
		a.ptr[i] = (scalar_t) i;
	}
	AlignedArray out(2);
	size_t reduce_size = 3;
	std::cout<<"before reduce_max"<<std::endl;
	ReduceSum(a, &out, reduce_size);
	for(size_t i = 0; i < 2; ++i){
		std::cout<<out.ptr[i]<<" ";
	}
	std::cout<<std::endl;
}

/**
 * Matmul
 */
void test11(){
	size_t sz = 6;
	AlignedArray a(sz);
	for(size_t i = 0; i < sz; ++i){
		a.ptr[i] = (scalar_t) i;
	}

	sz = 6;
	AlignedArray b(sz);
	for(size_t i = 0; i < sz; ++i){
		b.ptr[i] = (scalar_t) i;
	}

	sz = 4;
	AlignedArray c(sz);
	for(size_t i = 0; i < sz; ++i){
		c.ptr[i] = 0;
	}
	Matmul(a,b, &c, 2,3,2);
	for(size_t i = 0; i < sz; ++i){
		std::cout<<c.ptr[i]<<" ";
	}
	std::cout<<std::endl;


}

void test12(){
	size_t sz = 12;
	AlignedArray a(sz);
	for(size_t i = 0; i < sz; ++i){
		a.ptr[i] = (scalar_t) i;
	}

	sz = 20;
	AlignedArray b(sz);
	for(size_t i = 0; i < sz; ++i){
		b.ptr[i] = (scalar_t) i;
	}

	sz = 15;
	AlignedArray c(sz);
	for(size_t i = 0; i < sz; ++i){
		c.ptr[i] = 0;
	}
	Matmul(a,b, &c, 3,4,5);
	for(size_t i = 0; i < sz; ++i){
		std::cout<<c.ptr[i]<<" ";
	}
	std::cout<<std::endl;


}

inline void AlignedDot(const float* __restrict__ a,
                       const float* __restrict__ b,
                       float* __restrict__ out) {

  /**
   * Multiply together two TILE x TILE matrices, and _add _the result to out (it is important to add
   * the result to the existing out, which you should not set to zero beforehand).  We are including
   * the compiler flags here that enable the compile to properly use vector operators to implement
   * this function.  Specifically, the __restrict__ keyword indicates to the compile that a, b, and
   * out don't have any overlapping memory (which is necessary in order for vector operations to be
   * equivalent to their non-vectorized counterparts (imagine what could happen otherwise if a, b,
   * and out had overlapping memory).  Similarly the __builtin_assume_aligned keyword tells the
   * compiler that the input array will be aligned to the appropriate blocks in memory, which also
   * helps the compiler vectorize the code.
   *
   * Args:
   *   a: compact 2D array of size TILE x TILE
   *   b: compact 2D array of size TILE x TILE
   *   out: compact 2D array of size TILE x TILE to write to
   */

  a = (const float*)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
  b = (const float*)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
  out = (float*)__builtin_assume_aligned(out, TILE * ELEM_SIZE);

  /// BEGIN SOLUTION
  for(size_t i = 0 ; i < TILE; ++i){
  	  for(size_t j = 0; j < TILE; ++j){
  		 for(size_t k=0; k < TILE; ++k){
  			  out[i*TILE+j] += a[i*TILE+k] * b[k*TILE+j];
  			  //std::cout<<"out:"<<out[i*TILE+j]<<std::endl;
  		  }
  	  }
    }

  /// END SOLUTION
}

void MatmulTiled(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m,
                 uint32_t n, uint32_t p) {
  /**
   * Matrix multiplication on tiled representations of array.  In this setting, a, b, and out
   * are all *4D* compact arrays of the appropriate size, e.g. a is an array of size
   *   a[m/TILE][n/TILE][TILE][TILE]
   * You should do the multiplication tile-by-tile to improve performance of the array (i.e., this
   * function should call `AlignedDot()` implemented above).
   *
   * Note that this function will only be called when m, n, p are all multiples of TILE, so you can
   * assume that this division happens without any remainder.
   *
   * Args:
   *   a: compact 4D array of size m/TILE x n/TILE x TILE x TILE
   *   b: compact 4D array of size n/TILE x p/TILE x TILE x TILE
   *   out: compact 4D array of size m/TILE x p/TILE x TILE x TILE to write to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   *
   */
  /// BEGIN SOLUTION
	size_t M = m / TILE;
	size_t N = n / TILE;
	size_t P = p / TILE;
	// a : M x N tiles, b: N x P tiles, out: M x P tiles, each tile has TILE*TILE elemnts
	for (size_t I = 0 ; I < M; ++I){
		for (size_t J = 0; J < P; ++J){
			scalar_t *out_ptr = out->ptr + (I*P+J)*TILE*TILE; // OUT[I,J] I*P+J tiles
			memset((void*)out_ptr, 0, TILE*TILE * sizeof(scalar_t));
			for(size_t K = 0; K < N; ++K){
				scalar_t *a_ptr = a.ptr + (I*N+K)*TILE*TILE; // A[I,K] I*N+K tiles
				scalar_t *b_ptr = b.ptr + (K*P+J)*TILE*TILE; // B[K,J] K*P+J tiles
				AlignedDot(a_ptr, b_ptr, out_ptr);
			}
		}
	}
  /// END SOLUTION
}


// test AlignedDot
void test13(){
	size_t sz = 4;
	AlignedArray a(sz);
	AlignedArray b(sz);
	AlignedArray c(sz);
	for (size_t i = 0; i <sz; ++i){
		a.ptr[i] = i;
		b.ptr[i] = i;
		c.ptr[i] = 0;
	}
	AlignedDot(a.ptr, b.ptr, c.ptr);
	for(size_t i = 0; i < TILE; ++i){
		for(size_t j =0 ; j < TILE; ++j){
			std::cout<<c.ptr[i*TILE+j]<<" ";
		}
		std::cout<<std::endl;
	}
}

void test14(){
	size_t sz = 16;
	AlignedArray a(sz);
	std::vector<float> values(16);
	values = {0.0f, 1.0f, 4.0f, 5.0f, 2.0f, 3.0f, 6.0f, 7.0f, 8.0f, 9.0f, 12.0f, 13.0f, 10.0f, 11.0f, 14.0f, 15.0f};
	/**
	 * values is the tiling of:
	 * [ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15]
	 */

	AlignedArray b(sz);
	for(size_t i = 0; i < sz; ++i){
		a.ptr[i] = values[i];
		b.ptr[i] = a.ptr[i];
	}

	AlignedArray c(sz);
	for(size_t i = 0; i < sz; ++i){
		c.ptr[i] = 0;
	}
	size_t m=4, n= 4, p=4;
	MatmulTiled(a, b, &c, m, n, p);
	for (size_t i = 0; i < 4; ++i){
		for(size_t j = 0; j < 4; ++j){
			std::cout<<c.ptr[i*4+j]<<" ";
		}
		std::cout<<std::endl;
	}
	/*
	 * The expected output is:
	 * 56 62 152 174
	   68 74 196 218
       248 286 344 398
       324 362 452 506
       which is the tiling of
       [ 56,  62,  68,  74],
       [152, 174, 196, 218],
       [248, 286, 324, 362],
       [344, 398, 452, 506]
	 */

}


int main(int argc, char **argv){
	//std::cout<<"Hello world"<<std::endl;
	test1();
	std::cout<<"-----"<<std::endl;
	//test2();
	test3();
	std::cout<<"-----"<<std::endl;
	//test4();
	test5();
	//test6();
	//test7();
	//test8();
	//test9();
	//test10();
	//test11();
	//test12();
	//test13();
	//test14();
	return 0;
}
