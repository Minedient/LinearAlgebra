package util;

import objects.Matrix;
import objects.TwoMatrix;
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
import java.util.concurrent.BlockingDeque;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.LinkedBlockingDeque;

import static org.lwjgl.opencl.CL10.*;
import static org.lwjgl.opencl.CL10.clCreateContext;
import static org.lwjgl.system.MemoryUtil.NULL;
import static org.lwjgl.system.MemoryUtil.memUTF8;
import static util.IOUtil.ioResourceToByteBuffer;
import static util.InfoUtil.*;

public final class OpenCLWorker extends Thread {

    final ByteBuffer source;
    BlockingDeque<TwoMatrix> inputQueue = new LinkedBlockingDeque<>();
    BlockingDeque<Matrix> resultQueue = new LinkedBlockingDeque<>();
    private IntBuffer errorCodeRet;
    private long context;
    private long queue;
    private long matrixAMemory;
    private long matrixBMemory;
    private long matrixCMemory;
    private long program;
    private long kernel;

    {
        try {
            source = ioResourceToByteBuffer("MatrixMultiplication.cl", 4096);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public void sendTask(TwoMatrix task) {
        this.inputQueue.add(task);
    }

    private void initGPU() {

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

        final PointerBuffer strings = BufferUtils.createPointerBuffer(1);
        final PointerBuffer lengths = BufferUtils.createPointerBuffer(1);

        strings.put(0, source);
        lengths.put(0, source.remaining());

        program = clCreateProgramWithSource(context, strings, lengths, errorCodeRet);
        checkCLError(errorCodeRet);

        // To ensure the program is created before running, a latch is used.
        CountDownLatch latch = new CountDownLatch(1);
        final long finalDevice = device;
        checkCLError(clBuildProgram(program, device, "", CLProgramCallback.create((program1, userData) -> {
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
        kernel = clCreateKernel(program, "multiply", errorCodeRet);
        checkCLError(errorCodeRet);
    }

    private void killCL() {
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseMemObject(matrixAMemory);
        clReleaseMemObject(matrixBMemory);
        clReleaseMemObject(matrixCMemory);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        CL.destroy();
    }


    private Matrix sendDataToGPU(Matrix matrixA, Matrix matrixB) {
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

        clSetKernelArg1i(kernel, 0, matrixA.getNumOfRows());        // set M
        clSetKernelArg1i(kernel, 1, matrixB.getNumOfColumns());     // set N
        clSetKernelArg1i(kernel, 2, matrixA.getNumOfColumns());     // set K
        clSetKernelArg1p(kernel, 3, matrixAMemory);                 // set A    use 1p instead of 1d where p stands for pointer, not flowPoint, as I pass an ARRAY!@!!!!!
        clSetKernelArg1p(kernel, 4, matrixBMemory);                 // set B
        clSetKernelArg1p(kernel, 5, matrixCMemory);                 // set C

        clEnqueueNDRangeKernel(queue, kernel, 2, null, kernel2DGlobalWorkSize, null, null, null);
        clEnqueueReadBuffer(queue, matrixCMemory, false, 0, cMatrix, null, null);
        clFinish(queue);

        clReleaseMemObject(matrixAMemory);
        clReleaseMemObject(matrixBMemory);
        clReleaseMemObject(matrixCMemory);

        Matrix result = Matrix.createNewEmptyMatrix(matrixA.getNumOfRows(), matrixB.getNumOfColumns());
        result.setData(cMatrix);
        return result;
    }

    DoubleBuffer toDoubleBuffer(double[] doubles) {
        DoubleBuffer buffer = BufferUtils.createDoubleBuffer(doubles.length).put(doubles);
        buffer.rewind();
        return buffer;
    }

    @Override
    public void run() {
        // Initialize GPU
        initGPU();

        // waiting for work
        TwoMatrix input;
        while (!Thread.interrupted()) {
            while ((input = inputQueue.poll()) != null) {
                // send to gpu
                // receive answer and notify tasker
                resultQueue.add(sendDataToGPU(input.a(), input.b()));
            }
        }
        // release all objects
        killCL();
    }
}
