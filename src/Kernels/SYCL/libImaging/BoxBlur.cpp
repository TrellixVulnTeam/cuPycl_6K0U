#include <iostream>
#include <CL/sycl.hpp>


using namespace cl::sycl;


/*
I have to figure out how to make a device selector for SYCL.
Maybe add a string parameter to the function call irrespective of what kernel is called ???? 
Right Now it will just be the default selector
*/


void ImagingBoxBlur(Imaging* imOut, Imaging* imIn, float radius, int n){

}