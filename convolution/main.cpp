#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>
#include <assert.h>

#define __CL_ENABLE_EXCEPTIONS
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include "cl.hpp"



int main()
{
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::vector<cl::Kernel> kernels;

    try {

        // create platform
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        // create context
        cl::Context context(devices);

        // create command queue
        cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

        // load opencl source
        std::ifstream cl_file("convolution.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
                                                      cl_string.length() + 1));

        // create program
        cl::Program program(context, source);

        // compile opencl source
        try {
            program.build(devices);
        } catch(const cl::Error &err) {
            std::cerr
                    << "OpenCL compilation error" << std::endl
                    << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0])
                    << std::endl;
            throw err;
        }

        // create a message to send to kernel
        size_t M = 9, N = 1023;
        
        std::ifstream in("input.txt");
        std::ofstream out("output.txt");
        in >> N >> M;
        size_t old_N = N;
        size_t const block_size  = 16;
        if ((N % block_size) != 0) {
            N = (N / block_size + 1) * block_size;
        }
        
        std::vector<float> input(N * N, 0);
        std::vector<float> mask(M * M, 0);
        std::vector<float> output(N * N, 0);
        
        for (int i = 0; i < old_N; ++i)
            for (int k = 0; k < old_N; ++k) {
                in >> input[i * N + k];
            }
            
        for (int i = 0; i < M; ++i)
            for (int k = 0; k < M; ++k) {
                in >> mask[i * M + k];
            }


        // allocate device buffer to hold message
        cl::Buffer dev_input (context, CL_MEM_READ_ONLY, sizeof(float) * N * N);
        cl::Buffer dev_output(context, CL_MEM_WRITE_ONLY, sizeof(float) * N * N);
        cl::Buffer dev_mask  (context, CL_MEM_READ_ONLY, sizeof(float) * M * M);

        // copy from cpu to gpu
        queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(float) * N * N, &input[0]);
        queue.enqueueWriteBuffer(dev_mask, CL_TRUE, 0, sizeof(float)* M * M, &mask[0]);

        // load named kernel from opencl source
        queue.finish();
        cl::Kernel kernel_gmem(program, "gpu_convolution_gmem");
        cl::KernelFunctor convolution_gmem(kernel_gmem, queue, cl::NullRange, cl::NDRange(N,  N), cl::NDRange(block_size, block_size));
        cl::Event event = convolution_gmem(dev_input, dev_mask, dev_output, (int)M, (int)N);
        event.wait();


        queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(float) * N * N, &output[0]);
        out.setf(std::ios::fixed);
        for (int i = 0; i < old_N; ++i) {
            for (int k = 0; k < old_N; ++k) {
                out << std::setprecision(3) << output[i * N + k] << " ";
            }
            out << std::endl;
        }

    }
    catch (cl::Error e)
    {
        std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
    }

    return 0;
}
