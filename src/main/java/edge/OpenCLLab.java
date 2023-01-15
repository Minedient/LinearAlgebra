package edge;

import org.lwjgl.BufferUtils;
import org.lwjgl.PointerBuffer;
import util.InfoUtil;
import org.lwjgl.opencl.CL;
import org.lwjgl.opencl.CLContextCallback;
import org.lwjgl.opencl.CLProgramCallback;
import org.lwjgl.system.MemoryStack;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.concurrent.CountDownLatch;

import static util.InfoUtil.getProgramBuildInfoInt;
import static util.InfoUtil.getProgramBuildInfoStringASCII;
import static org.lwjgl.opencl.CL10.*;
import static org.lwjgl.opencl.CL10.clCreateContext;
import static org.lwjgl.system.MemoryUtil.NULL;
import static org.lwjgl.system.MemoryUtil.memUTF8;

class OpenCLLab {
    // The OpenCL kernel
    static final String source =
            "kernel void wtf(global const float *a, global const float *b, global float *answer) { "
                    + "  unsigned int xid = get_global_id(0); "
                    + "  answer[xid] = a[xid] * b[xid];"
                    + "}";

    // Data buffers to store the input and result data in
    static final FloatBuffer a = toFloatBuffer(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    static final FloatBuffer b = toFloatBuffer(new float[]{9, 8, 7, 6, 5, 4, 3, 2, 1, 0});

    static final FloatBuffer largerA = toFloatBuffer(fromOne());
    static final FloatBuffer largerB = toFloatBuffer(fromOne());
    static final FloatBuffer answer = BufferUtils.createFloatBuffer(largerA.capacity());
    private static final CLContextCallback CREATE_CONTEXT_CALLBACK = new CLContextCallback() {
        @Override
        public void invoke(long errinfo, long private_info, long cb, long user_data) {
            System.err.println("[LWJGL] cl_create_context_callback");
            System.err.println("\tInfo: " + memUTF8(errinfo));
        }
    };

    private static float[] fromOne() {
        float[] x = new float[256];
        for (int i = 0; i < 256; i++) {
            x[i] = i;
        }
        return x;
    }

    public static void main(String[] args) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            //test(stack);
            anotherTest(stack);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static void anotherTest(MemoryStack stack) throws Exception {

        //System.setProperty("org.lwjgl.opencl.explicitInit","true");

        IntBuffer pi = stack.mallocInt(1);
        checkCLError(clGetPlatformIDs(null, pi));
        if (pi.get(0) == 0) {
            throw new RuntimeException("No OpenCL platforms found.");
        }

        PointerBuffer platforms = stack.mallocPointer(pi.get(0));
        checkCLError(clGetPlatformIDs(platforms, (IntBuffer) null));
        System.out.println("Platform created");

        PointerBuffer ctxProps = stack.mallocPointer(3);
        ctxProps.put(0, CL_CONTEXT_PLATFORM).put(2, 0);
        System.out.println("CTX created");


        IntBuffer errcode_ret = stack.callocInt(1);
        System.out.println("ERRCODE created");

        long platform = 0;
        for (int p = 0; p < platforms.capacity(); p++) {
            platform = platforms.get(0);
            ctxProps.put(1, platform);
        }
        PointerBuffer devices = stack.mallocPointer(pi.get(0));
        checkCLError(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devices, (IntBuffer) null));

        long device = 0;
        for (int d = 0; d < devices.capacity(); d++) {
            device = devices.get(d);
        }

        // long context = clCreateContext(platform, devices, null, null, null);
        //long context = clCreateContext(ctxProps, device, CREATE_CONTEXT_CALLBACK, NULL, errcode_ret);
        long context = clCreateContext(ctxProps, device, CLContextCallback.create(
                (errinfo, private_info, cb, user_data) -> System.out.println(memUTF8(errinfo))
        ), NULL, errcode_ret);
        InfoUtil.checkCLError(errcode_ret);
        System.out.println("CONTEXT created");

        //CLCommandQueue queue = clCreateCommandQueue(context, devices.get(0), CL_QUEUE_PROFILING_ENABLE, null);
        long queue = clCreateCommandQueue(context, device, NULL, errcode_ret);
        System.out.println("Command Q created");

        // Allocate memory for our two input buffers and our result buffer
        long aMem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, a, errcode_ret);
        //long buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, 128, errcode_ret);
        System.out.println("A Buffer created");
        clEnqueueWriteBuffer(queue, aMem, true, 0, a, null, null);
        long bMem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, b, errcode_ret);
        System.out.println("B Buffer created");
        clEnqueueWriteBuffer(queue, bMem, true, 0, b, null, null);
        long answerMem = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, answer, errcode_ret);
        System.out.println("OUTPUT Buffer created");
        clFinish(queue);

        // Create our program and kernel
        long program = clCreateProgramWithSource(context, source, errcode_ret);
        InfoUtil.checkCLError(errcode_ret);

        CLProgramCallback buildCallback;
        CountDownLatch latch = new CountDownLatch(1);
        long finalDevice = device;
        int errcode = clBuildProgram(program, device, "", buildCallback = CLProgramCallback.create((program1, user_data) -> {
            System.out.println(String.format(
                    "The cl_program [0x%X] was built %s",
                    program1,
                    getProgramBuildInfoInt(program1, finalDevice, CL_PROGRAM_BUILD_STATUS) == CL_SUCCESS ? "successfully" : "unsuccessfully"
            ));
            String log = getProgramBuildInfoStringASCII(program1, finalDevice, CL_PROGRAM_BUILD_LOG);
            if (!log.isEmpty()) {
                System.out.println(String.format("BUILD LOG:\n----\n%s\n-----", log));
            }

            latch.countDown();
        }), NULL);
        checkCLError(errcode);

        // Make sure the program has been built before proceeding
        try {
            latch.await();
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }

        // sum has to match a kernel method name in the OpenCL source
        long kernel = clCreateKernel(program, "wtf", errcode_ret);
        InfoUtil.checkCLError(errcode_ret);
        System.out.println("KERNEL created");

        // Execution our kernel
        PointerBuffer kernel1DGlobalWorkSize = BufferUtils.createPointerBuffer(1);
        System.out.println("KERNEL work size created");
        // This controls the size of the work
        kernel1DGlobalWorkSize.put(0, largerB.capacity());
        System.out.println("KERNEL work size copied");

        clSetKernelArg1p(kernel, 0, aMem);
        clSetKernelArg1p(kernel, 1, bMem);
        clSetKernelArg1p(kernel, 2, answerMem);

        System.out.println("Args send to kernel");

        clEnqueueNDRangeKernel(queue, kernel, 1, null, kernel1DGlobalWorkSize, null, null, null);
        System.out.println("KERNEL queued created");


        // Read the results memory back into our result buffer
        clEnqueueReadBuffer(queue, answerMem, true, 0, answer, null, null);
        System.out.println("and output ... created");

        clFinish(queue);
        // Print the result memory
        print(a);
        System.out.println("*");
        print(b);
        System.out.println("=");
        print(answer);

        //remove the memory first
        clReleaseMemObject(aMem);
        clReleaseMemObject(bMem);

        // Try it again with a larger data;
        aMem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, largerA, errcode_ret);
        InfoUtil.checkCLError(errcode_ret);
        bMem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, largerB, errcode_ret);
        InfoUtil.checkCLError(errcode_ret);

        clSetKernelArg1p(kernel, 0, aMem);
        clSetKernelArg1p(kernel, 1, bMem);

        System.out.println("Args send to kernel");
        clEnqueueNDRangeKernel(queue, kernel, 1, null, kernel1DGlobalWorkSize, null, null, null);
        System.out.println("KERNEL queued created");


        // Read the results memory back into our result buffer
        clEnqueueReadBuffer(queue, answerMem, true, 0, answer, null, null);
        System.out.println("and output ... created");


        clFinish(queue);
        // Print the result memory
        print(largerA);
        System.out.println("*");
        print(largerB);
        System.out.println("=");
        print(answer);


        // Clean up OpenCL resources
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseMemObject(aMem);
        clReleaseMemObject(bMem);
        clReleaseMemObject(answerMem);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        CL.destroy();
    }


    /**
     * Utility method to convert float array to float buffer
     *
     * @param floats - the float array to convert
     * @return a float buffer containing the input float array
     */
    static FloatBuffer toFloatBuffer(float[] floats) {
        FloatBuffer buf = BufferUtils.createFloatBuffer(floats.length).put(floats);
        buf.rewind();
        return buf;
    }


    /**
     * Utility method to print a float buffer
     *
     * @param buffer - the float buffer to print to System.out
     */
    static void print(FloatBuffer buffer) {
        for (int i = 0; i < buffer.capacity(); i++) {
            System.out.print(buffer.get(i) + " ");
        }
        System.out.println("");
    }

    private static void test(MemoryStack memoryStack) {
        // Check OpenCL devices
        IntBuffer platformIds = memoryStack.mallocInt(1);
        checkCLError(clGetPlatformIDs(null, platformIds));
        if (platformIds.get(0) == 0)
            throw new RuntimeException("No OpenCL platforms found.");
    }

    private static void checkCLError(int errorCode) {
        if (errorCode != CL_SUCCESS) {
            throw new RuntimeException(String.format("OpenCL error [%d]", errorCode));
        }
    }

}
