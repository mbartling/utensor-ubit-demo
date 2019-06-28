//#include "test_helper.h"
//#include "models/qmul0.hpp"
//#include "models/qmul1.hpp"
//#include "models/qmul2.hpp"
//#include "models/qmul3.hpp"
//#include "models/qmul4.hpp"
//#include "models/deep_mlp.hpp"
#include "models/input_data.h"
#include "models/mnist_for_uTensor.hpp"
#include "uTensor/core/context.hpp"
#include "models/mnist_for_uTensor_weight.hpp"
#include <cmath>
Context ctx;

// Math functions go here TODO move me
//
template <typename U>
double meanAbsErr(Tensor* A, Tensor* B) {
  uint32_t size_A, size_B;
  size_A = A->getSize();
  size_B = B->getSize();
  if (A->getSize() != B->getSize()) {
    /* %lu is different for 64bit machines. Need to make this cross platform
    TensorShape shape_A, shape_B;
    shape_A = A->getShape();
    shape_B = B->getShape();
    printf("shape_A: ");
    for (auto s : shape_A) {
      printf("%lu, ", s);
    }
    printf("\n");
    printf("shape_B: ");
    for (auto s : shape_B) {
      printf("%lu, ", s);
    }
    printf("\n");
    printf("size A: %lu, size B: %lu\r\n", size_A, size_B);
    */
    ERR_EXIT("Test.meanAbsErr(): dimension mismatch\r\n");
  }

  const U* elemA = A->read<U>(0, 0);
  const U* elemB = B->read<U>(0, 0);

  double accm_err = 0.0;
  double total_size = (double) A->getSize();
  for (uint32_t i = 0; i < A->getSize(); i++) {
    accm_err += ((double)fabs((float)elemB[i] - (float)elemA[i]))/ total_size;
  }

  return accm_err;
}

// A being the reference
template <typename U>
double sumPercentErr(Tensor* A, Tensor* B) {
  uint32_t size_A, size_B;
  size_A = A->getSize();
  size_B = B->getSize();
  if (A->getSize() != B->getSize()) {
    
    ERR_EXIT("Test.sumPercentErr(): dimension mismatch\r\n");
  }


  double accm = 0.0;
  for (uint32_t i = 0; i < A->getSize(); i++) {
  const U* elemA = A->read<U>(i, 1);
  const U* elemB = B->read<U>(i, 1);
    if (elemA[0] != 0.0f) {
      accm += (double)fabs(((float)elemB[0] - (float)elemA[0]) / ((float)elemA[0]));
    } else {
      if (elemB[0] != 0) {
        accm += std::numeric_limits<float>::quiet_NaN();
      }
    }
  }
  return accm;
}
template<typename U>
double meanPercentErr(Tensor* A, Tensor* B) {
  double sum = sumPercentErr<U>(A, B);
  return sum / A->getSize();
}

template<typename U>
double sum(Tensor* input) {
  const U* elem = input->read<U>(0, 0);
  double accm = 0.0;
  for (uint32_t i = 0; i < input->getSize(); i++) {
    accm += (double)elem[i];
  }

  return accm;
}
template <typename T>
bool testshape(std::vector<T> src, std::vector<T> res, std::vector<uint8_t> permute) {
  bool pass = true;
  for (size_t i = 0; i < permute.size(); i++) {
    if (src[permute[i]] != res[i]) {
      pass = false;
      return pass;
    }
  }
  return pass;
}

bool testsize(uint32_t src, uint32_t res) {
  bool pass = true;
  if (src != res) {
      pass = false;
      return pass;
  }
  return pass;
}
template <typename T>
bool testval(T src, T res) {
  bool pass = true;
  if (src == res) {
    return pass;
  }
  return false;
}


//void test_operators_qmul0(void) {
//    ctx.gc();
//
//    get_qmul0_ctx(ctx); 
//    ctx.eval();
//    S_TENSOR res = ctx.get({"c_0:0"});
//    S_TENSOR ref = ctx.get({"ref_0:0"});
//
//    double result = meanPercentErr<float>(ref.get(), res.get());
////    EXPECT_EQ(result, 0);
//
//}
//
//void test_operators_qmul1(void) {
//    ctx.gc();
//
//    get_qmul1_ctx(ctx); 
//    ctx.eval();
//    S_TENSOR res = ctx.get({"c_1:0"});
//    S_TENSOR ref = ctx.get({"ref_1:0"});
//
//    double result = meanPercentErr<float>(ref.get(), res.get());
////    EXPECT_EQ(result, 0);
//
//}
//
//void test_operators_qmul2(void) {
//    ctx.gc();
//
//    get_qmul2_ctx(ctx); 
//    ctx.eval();
//    S_TENSOR res = ctx.get({"c_2:0"});
//    S_TENSOR ref = ctx.get({"ref_2:0"});
//
////    double result = meanPercentErr<float>(ref.get(), res.get());
////    EXPECT_EQ(result, 0);
//
//}
//
//void test_operators_qmul3(void) {
//    ctx.gc();
//
//    get_qmul3_ctx(ctx); 
//    ctx.eval();
//    S_TENSOR res = ctx.get({"c_3:0"});
//    S_TENSOR ref = ctx.get({"ref_3:0"});
//
////    double result = meanPercentErr<float>(ref.get(), res.get());
////    EXPECT_EQ(result, 0);
//
//}
//
//void test_operators_qmul4(void) {
//    ctx.gc();
//
//    get_qmul4_ctx(ctx); 
//    ctx.eval();
//    S_TENSOR res = ctx.get({"c_4:0"});
//    S_TENSOR ref = ctx.get({"ref_4:0"});
//
////    double result = meanPercentErr<float>(ref.get(), res.get());
////    EXPECT_EQ(result, 0);
//
//}

// void test_operators_mnist(void) {
//     Image<float>* img = new Image<float>(28, 28);
//     ctx.gc();
//     for(int i = 0 ; i < 28*28; i++){
//         (*img)[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
//     }
//     img->get_data()->resize({1, 784});
//     get_deep_mlp_ctx(ctx, img->get_data()); 
//     ctx.eval();
//     // S_TENSOR res = ctx.get({"c_4:0"});
//     // S_TENSOR ref = ctx.get({"ref_4:0"});

// //    double result = meanPercentErr<float>(ref.get(), res.get());
// //    EXPECT_EQ(result, 0);
//     delete img;
// }

void ubit() {
 ctx.gc();
//  Tensor* input_x = new WrappedRamTensor<float>({1, 784}, (float*) input_data);
 Tensor* input_x = new WrappedRamTensor<float>({1, 28, 28}, (float*) input_data);

  get_mnist_for_uTensor_ctx(ctx);  // pass the tensor to the context
  S_TENSOR pred_tensor = ctx.get(sref_model_mul_3_0);  // getting a reference to the output tensor
  ctx.eval(); //trigger the inference

  const float* pred_label = pred_tensor->read<float>(0, 0);  //getting the result back
  printf("Predicted label: %f\r\n", pred_label[0]);
  const float reference[] = {0.47058824, 0., 0., 0.28235295, 0., 0., 0., 0., 1.1058824, 0.8};
  float sum = 0;
  for(int i = 0; i < 9; i++){
    if(reference[i] != 0)
      sum += (reference[i] - pred_label[i])/reference[i];
  }
    sum  *= 100.0/9.0;
    printf("Mean percent error %f\n", sum);
}

// First configure the uTensor test runner
//  UTENSOR_TEST_CONFIGURE()
//  
//  UTENSOR_TEST(operators, qmul0, "Test qmul 0")
//  UTENSOR_TEST(operators, qmul1, "Test qmul 1")
//  UTENSOR_TEST(operators, qmul2, "Test qmul 2")
//  UTENSOR_TEST(operators, qmul3, "Test qmul 3")
//  UTENSOR_TEST(operators, qmul4, "Test qmul 4")
//  
//  
//  // Third, run like hell
//  UTENSOR_TEST_RUN()
//
int main() {

    printf("Starting test\n");
    // test_operators_qmul0();
    // test_operators_mnist();
    // test_operators_qmul1();
    ubit();


}
