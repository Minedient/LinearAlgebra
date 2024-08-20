package objects;

import org.lwjgl.BufferUtils;
import org.lwjgl.PointerBuffer;
import org.lwjgl.opencl.CL;
import org.lwjgl.opencl.CLContextCallback;
import org.lwjgl.opencl.CLProgramCallback;
import org.lwjgl.system.MemoryStack;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;

import static org.lwjgl.opencl.CL10.*;
import static org.lwjgl.opencl.CL10.clCreateCommandQueue;
import static org.lwjgl.system.MemoryUtil.NULL;
import static org.lwjgl.system.MemoryUtil.memUTF8;
import static util.IOUtil.ioResourceToByteBuffer;
import static util.InfoUtil.*;

public class OpenCLInteractor {

    private boolean initialized = false;
    final List<ByteBuffer> sources = new ArrayList<>();
    private IntBuffer errorCodeRet;
    private long context;
    private long queue;
    final List<Long> programs = new ArrayList<>();
    final List<Long> kernels = new ArrayList<>();
    final List<String> kernelsName = new ArrayList<>();

    private long matrixAMemory;
    private long matrixBMemory;
    private long matrixCMemory;
    private long matrixDMemory;


    private void initializeSources(){
        try {
            sources.add(ioResourceToByteBuffer("MatrixMultiplication.cl", 4096));
            sources.add(ioResourceToByteBuffer("ForwardPass.cl", 4096));
            kernelsName.add("multiply");
            kernelsName.add("forwardPass");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public void initialize(){
        initializeSources();

        MemoryStack memoryStack = MemoryStack.stackPush();
        // Get the id of the OpenCL platform
        IntBuffer platformID = memoryStack.mallocInt(1);
        checkCLError(clGetPlatformIDs(null, platformID));
        if (platformID.get(0) == 0) {
            throw new RuntimeException("No OpenCL platforms found.");
        }
        PointerBuffer platforms = memoryStack.mallocPointer(platformID.get(0));
        checkCLError(clGetPlatformIDs(platforms, (IntBuffer) null));

        PointerBuffer ctxProps = memoryStack.mallocPointer(3);
        ctxProps.put(0, CL_CONTEXT_PLATFORM).put(2, 0);

        errorCodeRet = memoryStack.callocInt(1);

        // Get platform (assume one only)
        long platform = platforms.get(0);
        ctxProps.put(1, platform);
        checkCLError(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, null, platformID));

        PointerBuffer devices = memoryStack.mallocPointer(platformID.get(0));
        checkCLError(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devices, (IntBuffer) null));

        long device = devices.get(0);
        context = clCreateContext(ctxProps, device, CLContextCallback.create((errorInfo, privateInfo, cb, userData) -> System.out.println(memUTF8(errorInfo))), NULL, errorCodeRet);
        checkCLError(errorCodeRet);

        queue = clCreateCommandQueue(context, device, NULL, errorCodeRet);

        for (int i = 0; i < sources.size(); i++) {
            final PointerBuffer strings = BufferUtils.createPointerBuffer(1);
            final PointerBuffer lengths = BufferUtils.createPointerBuffer(1);

            strings.put(0, sources.get(i));
            lengths.put(0, sources.get(i).remaining());
            programs.add(clCreateProgramWithSource(context, strings, lengths, errorCodeRet));
            checkCLError(errorCodeRet);

            // To ensure the program is created before running, a latch is used.
            CountDownLatch latch = new CountDownLatch(1);
            final long finalDevice = device;
            checkCLError(clBuildProgram(programs.get(i), device, "", CLProgramCallback.create((program1, userData) -> {
                System.out.println(String.format(
                        "The cl_program [0x%X] was built %s",
                        program1,
                        getProgramBuildInfoInt(program1, finalDevice, CL_PROGRAM_BUILD_STATUS) == CL_SUCCESS ? "successfully" : "unsuccessfully"
                ));
                String log = getProgramBuildInfoStringASCII(program1, finalDevice, CL_PROGRAM_BUILD_LOG);
                if (!log.isEmpty()) {
                    System.out.println(String.format("BUILD LOG:%n----%n%s%n-----", log));
                }

                latch.countDown();
            }), NULL));

            // Make sure the program has been built before proceeding
            try {
                latch.await();
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }

            kernels.add(clCreateKernel(programs.get(i), kernelsName.get(i), errorCodeRet));
            checkCLError(errorCodeRet);
        }
        initialized = true;
    }

    public void exit(){
        initialized = false;
        for(Long kernel:kernels)
            clReleaseKernel(kernel);
        for(Long program:programs)
            clReleaseProgram(program);
        clReleaseMemObject(matrixAMemory);
        clReleaseMemObject(matrixBMemory);
        clReleaseMemObject(matrixCMemory);
        clReleaseMemObject(matrixDMemory);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        CL.destroy();
    }

    public Matrix clMultiply(Matrix matrixA, Matrix matrixB){
        if(!initialized)
            throw new OpenCLNotInitializedException();
        final PointerBuffer kernel2DGlobalWorkSize = BufferUtils.createPointerBuffer(2);
        kernel2DGlobalWorkSize.put(0, matrixA.getNumOfRows()).put(1, matrixB.getNumOfColumns());
        checkCLError(errorCodeRet);

        DoubleBuffer aMatrix = toDoubleBuffer(matrixA.getData());
        DoubleBuffer bMatrix = toDoubleBuffer(matrixB.getData());
        DoubleBuffer cMatrix = BufferUtils.createDoubleBuffer(matrixA.getNumOfRows() * matrixB.getNumOfColumns());

        matrixAMemory = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, aMatrix, errorCodeRet);
        checkCLError(errorCodeRet);
        matrixBMemory = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bMatrix, errorCodeRet);
        checkCLError(errorCodeRet);
        matrixCMemory = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, cMatrix, errorCodeRet);
        checkCLError(errorCodeRet);
        // Finish the queue
        clEnqueueWriteBuffer(queue, matrixAMemory, true, 0, aMatrix, null, null);
        clEnqueueWriteBuffer(queue, matrixBMemory, true, 0, bMatrix, null, null);
        clFinish(queue);

        clSetKernelArg1i(kernels.get(0), 0, matrixA.getNumOfRows());        // set M
        clSetKernelArg1i(kernels.get(0), 1, matrixB.getNumOfColumns());     // set N
        clSetKernelArg1i(kernels.get(0), 2, matrixA.getNumOfColumns());     // set K
        clSetKernelArg1p(kernels.get(0), 3, matrixAMemory);                 // set A    use 1p instead of 1d where p stands for pointer, not flowPoint, as I pass an ARRAY!@!!!!!
        clSetKernelArg1p(kernels.get(0), 4, matrixBMemory);                 // set B
        clSetKernelArg1p(kernels.get(0), 5, matrixCMemory);                 // set C

        clEnqueueNDRangeKernel(queue, kernels.get(0), 2, null, kernel2DGlobalWorkSize, null, null, null);
        clEnqueueReadBuffer(queue, matrixCMemory, false, 0, cMatrix, null, null);
        clFinish(queue);

        clReleaseMemObject(matrixAMemory);
        clReleaseMemObject(matrixBMemory);
        clReleaseMemObject(matrixCMemory);

        Matrix result = Matrix.createNewEmptyMatrix(matrixA.getNumOfRows(), matrixB.getNumOfColumns());
        result.setData(cMatrix);

        return result;
    }

    public Matrix clForwardPass(final Matrix weights, final Matrix inputs, final Matrix bias, final int mode){
        if(!initialized)
            throw new OpenCLNotInitializedException();
        final PointerBuffer kernel2DGlobalWorkSize = BufferUtils.createPointerBuffer(2);
        kernel2DGlobalWorkSize.put(0, weights.getNumOfRows()).put(1, inputs.getNumOfColumns());
        checkCLError(errorCodeRet);

        DoubleBuffer weightsMatrix = toDoubleBuffer(weights.getData());
        DoubleBuffer inputsMatrix = toDoubleBuffer(inputs.getData());
        DoubleBuffer biasMatrix = toDoubleBuffer(bias.getData());
        DoubleBuffer resultMatrix = BufferUtils.createDoubleBuffer(bias.getColumnSize());

        matrixAMemory = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weightsMatrix, errorCodeRet);
        checkCLError(errorCodeRet);
        matrixBMemory = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, inputsMatrix, errorCodeRet);
        checkCLError(errorCodeRet);
        matrixCMemory = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, biasMatrix, errorCodeRet);
        checkCLError(errorCodeRet);
        matrixDMemory = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, resultMatrix, errorCodeRet);
        checkCLError(errorCodeRet);
        // Finish the queue
        clEnqueueWriteBuffer(queue, matrixAMemory, true, 0, weightsMatrix, null, null);
        clEnqueueWriteBuffer(queue, matrixBMemory, true, 0, inputsMatrix, null, null);
        clEnqueueWriteBuffer(queue, matrixCMemory, true, 0, biasMatrix, null, null);
        clFinish(queue);

        clSetKernelArg1i(kernels.get(1), 0, weights.getNumOfRows());        // set M
        clSetKernelArg1i(kernels.get(1), 1, inputs.getNumOfColumns());      // set N
        clSetKernelArg1i(kernels.get(1), 2, weights.getNumOfColumns());     // set K
        clSetKernelArg1p(kernels.get(1), 3, matrixAMemory);                 // set weightMatrix    use 1p instead of 1d where p stands for pointer, not flowPoint, as I pass an ARRAY!@!!!!!
        clSetKernelArg1p(kernels.get(1), 4, matrixBMemory);                 // set inputsMatrix
        clSetKernelArg1p(kernels.get(1), 5, matrixCMemory);                 // set biasMatrix
        clSetKernelArg1p(kernels.get(1), 6, matrixDMemory);                 // set resultsMatrix
        clSetKernelArg1i(kernels.get(1), 7, mode);                          // set MODE

        clEnqueueNDRangeKernel(queue, kernels.get(1), 2, null, kernel2DGlobalWorkSize, null, null, null);
        clEnqueueReadBuffer(queue, matrixDMemory, false, 0, resultMatrix, null, null);
        clFinish(queue);

        clReleaseMemObject(matrixAMemory);
        clReleaseMemObject(matrixBMemory);
        clReleaseMemObject(matrixCMemory);
        clReleaseMemObject(matrixDMemory);

        Matrix result = Matrix.createNewEmptyMatrix(weights.getNumOfRows(), inputs.getNumOfColumns());
        result.setData(resultMatrix);

        return result;
    }

    private DoubleBuffer toDoubleBuffer(double[] doubles) {
        DoubleBuffer buffer = BufferUtils.createDoubleBuffer(doubles.length).put(doubles);
        buffer.rewind();
        return buffer;
    }

    private static class OpenCLNotInitializedException extends RuntimeException {
        @Override
        public String getMessage() {
            return super.getMessage();
        }
    }
}
