#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <stdio.h>
namespace py = pybind11;
using namespace std;
/**
 * @x size of m by n, already allocated outside
 * @mat the returned matrix, it could be either allocated outside (if passed in mat is not NULL), or allocated inside (if passed in mat is NULL)
 * @return 2d matrix that is backed up by x
*/
float** vec_to_mat(float *x, float **mat, size_t m, size_t n){
    if(mat == NULL){
        mat = (float **) malloc(sizeof(float *) * m);
    }
    for(size_t i = 0; i < m; ++i){
        mat[i] = &(x[i*n]);
    }
    return mat;
}

/**
 * free 2d matrix
*/
void free_mat(float **mat, size_t m, size_t n){
    /*for(size_t i = 0; i < m; ++i){
        free(mat[i]); // weiz 2023-12-18, I didn't allocate mat[i], so no need to free
    }*/
    free(mat);
}

/**
 * Z = XY 
 * X: m by n
 * Y: n by p
 * Z: m by p
*/
void matmul(float **X, float **Y, float **Z, size_t m, size_t n, size_t p){
    for (size_t i=0; i < m; ++i){
        for (size_t j = 0; j < p;++j){
            Z[i][j] = 0.0f;
            for(size_t k = 0; k < n; ++k){
                Z[i][j] += X[i][k] * Y[k][j];
            }
        }
    }
}

void softmax(float **Z, float **A, size_t m, size_t n){
    for (size_t i = 0; i < m; ++i){
        float row_sum = 0;

        for(size_t j =0 ; j < n; ++j){
            A[i][j] = exp(Z[i][j]);
            row_sum += A[i][j];
            //std::cout<<i<<","<<j<<std::endl;
        }
        for(int j=n-1; j>=0; --j){ // weiz: it has to be int not size_t because size_t cannot hold negative number
            A[i][j] /= row_sum;
            //std::cout<<"!!!"<<i<<","<<j<<std::endl;
        }
    }
}

/**
 * A: predicted probability distribution, m by k, m is batch size, k is the class size
 * y: ground truth labels, size of m, 
 * k : number of classes
 * return: averge loss across batch size
*/
float cross_entropy_loss(float **A, const unsigned char *y, size_t m, size_t k){
    float total_loss = 0.0f;
    for(size_t i = 0; i < m; ++i){
        total_loss += (log(A[i][y[i]]) * (-1));
    }
    return total_loss / m;
}
/**
 * Grad = X^T matmul Y
 * X: m by n
 * Y: m by k
 * G: n by k
 * m : batch size
 * n: feature size
 * k: class size
*/
void X_transpose_matmul_Y(float **X, float **Y, float **Grad, size_t m, size_t n, size_t k){
    for(size_t i = 0 ; i < n; ++i){
        for (size_t j = 0; j < k; ++j){
            Grad[i][j] = 0.0f;
            for (size_t s = 0; s < m; ++s){
                Grad[i][j] += X[s][i]* Y[s][j]; // TO CHECK!!
            }
        }
    }
}

/** below is the cpp interface with python weiz 2023-12-17 **/
void matmul_cpp(float *_X, float *_Y, float *_Z, size_t m, size_t n, size_t p){
     float **X = vec_to_mat(_X, NULL, m, n);
     float **Y = vec_to_mat(_Y, NULL, n, p);
     float **Z = vec_to_mat(_Z, NULL, m, p);
     matmul(X, Y, Z, m, n, p);
}

void softmax_cpp(float *_Z, float * _A, size_t m, size_t k){
     float **Z = vec_to_mat(_Z, NULL, m, k);
     float **A = vec_to_mat(_A, NULL, m, k);
     softmax(Z, A, m, k);
}

float cross_entropy_loss_cpp(float *_A , const unsigned char *_y, size_t m, size_t k){
     float **A = vec_to_mat(_A, NULL, m, k);
     return cross_entropy_loss(A, _y, m, k);
}

void X_transpose_matmul_Y_cpp(float *_X, float *_Y, float *_Z, size_t m, size_t n, size_t k){
    float **X = vec_to_mat(_X, NULL, m, n);
    float **Y = vec_to_mat(_Y, NULL, m, k);
    float **Z = vec_to_mat(_Z, NULL, n, k);
    X_transpose_matmul_Y(X, Y, Z, m, n, k);
}



void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    // create intermediate memory buffers
    //std::cout<<"before allocating memory"<<std::endl;
    float *_Z = (float *) malloc(sizeof(float)*batch*k);
    float **Z = vec_to_mat(_Z, NULL, batch , k);
    float *_A = (float *) malloc(sizeof(float)*batch*k);
    float **A = vec_to_mat(_A, NULL, batch , k);
    float ** X_batch = (float **) malloc(sizeof(float *) * batch);
    X_batch = vec_to_mat((float *)X, X_batch, batch, n);
    float **theta_mat = vec_to_mat(theta, NULL, n, k);
    float *_grad = (float *) malloc(sizeof(float) * n * k);
    float **grad = vec_to_mat(_grad, NULL, n, k);
    unsigned int start_idx = 0;
    unsigned int end_idx = start_idx + batch;
    //std::cout<<"after allocating memory"<<std::endl;
    while (start_idx < m){
       
        // step1: fwd prop
        matmul(X_batch, theta_mat, Z, batch, n, k);
        softmax(Z, A, batch, k);
        //float avg_loss = cross_entropy_loss(A, y, batch, k);
        //printf("avg loss %f \n", avg_loss);
        

        // step2: bwd prop
        for(size_t i = 0; i < batch; ++i){
            A[i][y[i]] -= 1.0f;
        }
        X_transpose_matmul_Y(X_batch, A, grad, batch, n, k);
        for(size_t i = 0; i < n;++i){
            for(size_t j = 0; j < k; ++j){
                theta_mat[i][j] -= ((lr)*(grad[i][j]/batch));
            }
        }

        // step3: get next batch
        start_idx += batch; // start_idx is really per sample based
        end_idx = ((end_idx + batch)>=m) ? m:(end_idx+batch);
        batch = end_idx - start_idx;
        X_batch = vec_to_mat((float *)(X+start_idx*n), X_batch, batch, n); // weiz 2023-12-18, the bug that I had: i had X+start_idx, instead of X+start_idx*n
        y += batch;

    }

    // cleanup memory allocation
    free_mat(Z, batch, k);
    free(_Z);
    free_mat(A, batch, k);
    free(_A);
    free_mat(grad, n, k);
    free(_grad);
    free_mat(X_batch, batch, n);
    free_mat(theta_mat, n, k);
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));

     m.def("matmul_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<float, py::array::c_style> Y,
           py::array_t<float, py::array::c_style> Z
          ) {
        matmul_cpp(
        	static_cast<float*>(X.request().ptr),
            static_cast<float*>(Y.request().ptr),
            static_cast<float*>(Z.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            Z.request().shape[1]
           );
    },
    py::arg("X"), py::arg("Y"), py::arg("Z"));

    m.def("softmax_cpp",
    	[](py::array_t<float, py::array::c_style> Z,
           py::array_t<float, py::array::c_style> A
          ) {
        softmax_cpp(
        	static_cast<float*>(Z.request().ptr),
            static_cast<float*>(A.request().ptr),
            Z.request().shape[0],
            A.request().shape[1]
           );
    },
    py::arg("Z"), py::arg("A"));

    m.def("cross_entropy_loss_cpp",
    	[](py::array_t<float, py::array::c_style> A,
    	   py::array_t<unsigned char, py::array::c_style> y
          ) {
        return cross_entropy_loss_cpp(
        	static_cast<float*>(A.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            A.request().shape[0],
            A.request().shape[1]
           );
    },
    py::arg("A"), py::arg("y"));

    m.def("X_transpose_matmul_Y_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<float, py::array::c_style> Y,
           py::array_t<float, py::array::c_style> Z
          ) {
        X_transpose_matmul_Y_cpp(
        	static_cast<float*>(X.request().ptr),
            static_cast<float*>(Y.request().ptr),
            static_cast<float*>(Z.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            Z.request().shape[1]
           );
    },
    py::arg("X"), py::arg("Y"), py::arg("Z"));
}
