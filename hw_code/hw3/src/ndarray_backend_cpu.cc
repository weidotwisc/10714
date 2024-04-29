#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace needle {
namespace cpu {

#define ALIGNMENT 256
#define TILE 8
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



void Fill(AlignedArray* out, scalar_t val) {
  /**
   * Fill the values of an aligned array with val
   */
  for (int i = 0; i < out->size; i++) {
    out->ptr[i] = val;
  }
}

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
		std::vector<int32_t> & repr){
	assert(repr.size() == src_strides.size());
	assert(repr.size() == dst_strides.size());
	size_t src_ptr_offset = 0;
	size_t dst_ptr_offset = 0;
	for (size_t i = 0; i < repr.size(); ++i){
		src_ptr_offset += repr[i]*src_strides[i];
		dst_ptr_offset += repr[i]*dst_strides[i];
	}
	*(dst_ptr+dst_ptr_offset) = *(src_ptr+src_ptr_offset);
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
		std::vector<int32_t> shape
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
		fill_tensor_at(real_dest_ptr, real_src_ptr, dest_strides, src_strides, repr);
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
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
   *         product of items in shape, but convenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */

  /// BEGIN SOLUTION
  assert(false && "Not Implemented");
  /// END SOLUTION
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
  assert(false && "Not Implemented");
  /// END SOLUTION
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
  assert(false && "Not Implemented");
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
  assert(false && "Not Implemented");
  /// END SOLUTION
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
  assert(false && "Not Implemented");
  /// END SOLUTION
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
  assert(false && "Not Implemented");
  /// END SOLUTION
}

}  // namespace cpu
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cpu, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cpu;

  m.attr("__device_name__") = "cpu";
  m.attr("__tile_size__") = TILE;

  py::class_<AlignedArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def("ptr", &AlignedArray::ptr_as_int)
      .def_readonly("size", &AlignedArray::size);

  // return numpy array (with copying for simplicity, otherwise garbage
  // collection is a pain)
  m.def("to_numpy", [](const AlignedArray& a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });
    return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset);
  });

  // convert from numpy (with copying)
  m.def("from_numpy", [](py::array_t<scalar_t> a, AlignedArray* out) {
    std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE);
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  // m.def("ewise_mul", EwiseMul);
  // m.def("scalar_mul", ScalarMul);
  // m.def("ewise_div", EwiseDiv);
  // m.def("scalar_div", ScalarDiv);
  // m.def("scalar_power", ScalarPower);

  // m.def("ewise_maximum", EwiseMaximum);
  // m.def("scalar_maximum", ScalarMaximum);
  // m.def("ewise_eq", EwiseEq);
  // m.def("scalar_eq", ScalarEq);
  // m.def("ewise_ge", EwiseGe);
  // m.def("scalar_ge", ScalarGe);

  // m.def("ewise_log", EwiseLog);
  // m.def("ewise_exp", EwiseExp);
  // m.def("ewise_tanh", EwiseTanh);

  // m.def("matmul", Matmul);
  // m.def("matmul_tiled", MatmulTiled);

  // m.def("reduce_max", ReduceMax);
  // m.def("reduce_sum", ReduceSum);
}
