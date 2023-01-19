import exceptions.MatrixDimensionsNotMatchException;
import objects.LUMatrixGroup;
import objects.Matrix;
import objects.OpenCLInteractor;
import org.junit.jupiter.api.Test;
import org.lwjgl.BufferUtils;
import org.lwjgl.PointerBuffer;
import util.InfoUtil;
import util.IOUtil;
import org.lwjgl.opencl.CL;
import org.lwjgl.opencl.CLContextCallback;
import org.lwjgl.opencl.CLProgramCallback;
import org.lwjgl.system.MemoryStack;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.IntBuffer;
import java.util.Arrays;
import java.util.concurrent.CountDownLatch;

import static org.junit.jupiter.api.Assertions.*;
import static util.InfoUtil.*;
import static org.lwjgl.opencl.CL10.*;
import static org.lwjgl.opencl.CL10.CL_DEVICE_TYPE_ALL;
import static org.lwjgl.system.MemoryUtil.NULL;
import static org.lwjgl.system.MemoryUtil.memUTF8;

class MatrixTest {


    private double[] randomizeDoubleArray(int size){
        double[] a = new double[size];
        for (int i = 0; i < a.length; i++)
            a[i] = Math.random();
        return a;
    }
    @Test
    void creation(){
        Matrix m1 = Matrix.createNewEmptyMatrix(3,2);
        Matrix m2 = Matrix.copyingMatrix(m1);
        Matrix m3 = Matrix.createNewEmptyMatrix(m1.getDimension());

        assertSame(3, m1.getNumOfRows());
        assertSame(2, m1.getNumOfColumns());

        assertSame(3, m2.getNumOfRows());
        assertSame(2, m2.getNumOfColumns());

        assertSame(3, m3.getNumOfRows());
        assertSame(2, m3.getNumOfColumns());

        Matrix m4 = Matrix.createNewFilledMatrix(new double[]{1,2,3});
        System.out.println(m4);
    }

    @Test
    void transposition(){
        Matrix matrix = Matrix.createNewEmptyMatrix(4,2);
        assertSame(4, matrix.getNumOfRows());
        assertSame(2, matrix.getNumOfColumns());
        Matrix transposed = Matrix.transpose(matrix);
        assertSame(2, transposed.getNumOfRows());
        assertSame(4, transposed.getNumOfColumns());
    }

    @Test
    void numericalTests(){
        Matrix matrix = Matrix.createNewFilledMatrix(new double[]{1,2,3},new double[]{4,5,6},new double[]{7,8,9});
        System.out.println("Original:");
        System.out.println(matrix);
        Matrix transposed = Matrix.transpose(matrix);
        System.out.println("Transposed:");
        System.out.println(transposed);
        System.out.println("Set data on row 2 to [4,5,2] on original matrix:");
        matrix.setRow(1, new double[]{4,5,2});
        System.out.println("Result:");
        System.out.println(matrix);
        System.out.println("Set data on row 3 to [-3,7,1] on transposed matrix:");
        transposed.setColumn(2, new double[]{-3,7,1});
        System.out.println("Result:");
        System.out.println(transposed);
        assertSame(Matrix.createNewFilledMatrix(new double[]{1,2,3},new double[]{4,5,2},new double[]{7,8,9}), matrix);
        assertSame(Matrix.createNewFilledMatrix(new double[]{1,4,-3},new double[]{2,5,7},new double[]{3,6,1}), transposed);

    }

    @Test
    void getTests(){
        Matrix matrix = Matrix.createNewFilledMatrix(new double[]{1,2,3},new double[]{4,5,6},new double[]{7,8,9});

        System.out.println(Arrays.toString(matrix.getRow(0)));
        System.out.println(Arrays.toString(matrix.getColumn(0)));
        assertEquals(2.0, matrix.getRow(0)[1]);
        assertEquals(4.0, matrix.getColumn(0)[1]);

    }

    @Test
    void luDecompositionTest(){
        Matrix matrix = Matrix.createNewFilledMatrix(new double[]{1,2,3},new double[]{4,5,6},new double[]{7,8,8});
        LUMatrixGroup lu = Matrix.luDecomposition(matrix);
        System.out.println("Original matrix:");
        System.out.println(matrix);
        System.out.println("L Matrix");
        System.out.println(lu.l());
        System.out.println("U Matrix");
        System.out.println(lu.u());
    }

    @Test
    void determinantTest() throws MatrixDimensionsNotMatchException {
        Matrix matrix = Matrix.createNewFilledMatrix(new double[]{1,2,3},new double[]{4,5,6},new double[]{7,8,8});
        System.out.println("Original matrix:");
        System.out.println(matrix);
        System.out.println("Determinant: " + Matrix.determinant(matrix));
    }

    @Test
    void multiplicationTest() throws MatrixDimensionsNotMatchException {
        Matrix matrix = Matrix.createNewFilledMatrix(new double[]{1,2,3},new double[]{4,5,6},new double[]{7,8,8});
        LUMatrixGroup lu = Matrix.luDecomposition(matrix);
        System.out.println("Original matrix:");
        System.out.println(matrix);
        System.out.println("L Matrix");
        System.out.println(lu.l());
        System.out.println("U Matrix");
        System.out.println(lu.u());
        System.out.println("L x U:");
        System.out.println(Matrix.multiplication(lu.l(),lu.u()));
    }

    @Test
    void multiplicationTest2() throws MatrixDimensionsNotMatchException {
        Matrix matrix1 = Matrix.createNewFilledMatrix(2,2, new double[]{ 11,3,7,11 });
        Matrix matrix2 = Matrix.createNewFilledMatrix(2,3, new double[]{ 8,0,1,0,3,5 });
        Matrix answer = Matrix.createNewFilledMatrix(2,3, new double[]{ 88,9,26,56,33,62 });
        System.out.println("Matrix A:");
        System.out.println(matrix1);
        System.out.println("Matrix B:");
        System.out.println(matrix2);
        System.out.println("A x B:");
        System.out.println(Matrix.multiplication(matrix1, matrix2));
        assertEquals(answer, Matrix.multiplication(matrix1, matrix2));
    }

    @Test
    void inverseTest() throws MatrixDimensionsNotMatchException {
        Matrix matrix = Matrix.createNewFilledMatrix(new double[]{1,2,3},new double[]{4,5,6},new double[]{7,8,8});
        Matrix inverse = Matrix.inverse(matrix);
        System.out.println("Original matrix:");
        System.out.println(matrix);
        System.out.println("Inverse:");
        System.out.println(inverse);
        System.out.println("A x A*");
        System.out.println(Matrix.multiplication(matrix,inverse));
        assertEquals(Matrix.createNewFilledMatrix(new double[]{1,0,0},new double[]{0,1,0},new double[]{0,0,1}), Matrix.multiplication(matrix,inverse));
    }

    @Test
    void equalTest(){
        Matrix matrix = Matrix.createNewFilledMatrix(new double[]{1,2,3},new double[]{4,5,6},new double[]{7,8,8});
        Matrix matrix2 = Matrix.createNewFilledMatrix(new double[]{1,2,3},new double[]{4,5,6},new double[]{7,8,8});
        assertEquals(matrix2,matrix);
    }

    @Test
    void vectorMultiplicationTest() throws MatrixDimensionsNotMatchException {
        Matrix matrix = Matrix.createNewFilledMatrix(new double[]{2,-3}, new double[]{-1,4});
        Matrix vector = Matrix.createNewEmptyColumnVector(2);
        vector.setColumn(0, new double[]{1, -1});
        System.out.println("Original matrix:");
        System.out.println(matrix);
        System.out.println("Original vector:");
        System.out.println(vector);
        System.out.println("Answer:");
        System.out.println(Matrix.multiplication(matrix,vector));
    }


    @Test
    void singleThreadedMultiplicationSpeed(){
        Matrix a = Matrix.createNewEmptyMatrix(2000,2000).fillRandomDoubles();
        Matrix b = Matrix.createNewEmptyMatrix(2000,2000).fillRandomDoubles();

        long start = System.currentTimeMillis();
        for (int i = 0; i < 10; i++) {
            System.out.println("Doing " + (i+1) +"(th) matrix multiplication");
            try {
                Matrix.multiplication(a,b);
            } catch (MatrixDimensionsNotMatchException e) {
                throw new RuntimeException(e);
            }
        }
        System.out.println("Average time taken: " + (System.currentTimeMillis() - start) / 10000.0 + " seconds");
    }

    @Test
    void multiThreadedMultiplicationSpeed(){
        Matrix a = Matrix.createNewEmptyMatrix(2000,2000).fillRandomDoubles();
        Matrix b = Matrix.createNewEmptyMatrix(2000,2000).fillRandomDoubles();

        long start = System.currentTimeMillis();
        for (int i = 0; i < 10; i++) {
            System.out.println("Doing " + (i+1) +"(th) matrix multiplication");
            try {
                Matrix.multiThreadedMultiplication(a,b);
            } catch (MatrixDimensionsNotMatchException e) {
                throw new RuntimeException(e);
            }
        }
        System.out.println("Average time taken: " + (System.currentTimeMillis() - start) / 10000.0 + " seconds");
    }


    @Test
    void multiThreadedMultiplicationValueTest() throws MatrixDimensionsNotMatchException {
        Matrix matrix = Matrix.createNewFilledMatrix(new double[]{2,-3}, new double[]{-1,4});
        Matrix vector = Matrix.createNewEmptyColumnVector(2);
        vector.setColumn(0, new double[]{1, -1});
        System.out.println("Original matrix:");
        System.out.println(matrix);
        System.out.println("Original vector:");
        System.out.println(vector);
        System.out.println("Answer:");
        System.out.println(Matrix.multiThreadedMultiplication(matrix,vector));

        Matrix matrix1 = Matrix.createNewFilledMatrix(2,2, new double[]{ 11,3,7,11 });
        Matrix matrix2 = Matrix.createNewFilledMatrix(2,3, new double[]{ 8,0,1,0,3,5 });
        Matrix answer = Matrix.createNewFilledMatrix(2,3, new double[]{ 88,9,26,56,33,62 });
        System.out.println("Matrix A:");
        System.out.println(matrix1);
        System.out.println("Matrix B:");
        System.out.println(matrix2);
        System.out.println("A x B:");
        System.out.println(Matrix.multiThreadedMultiplication(matrix1, matrix2));
        assertEquals(answer, Matrix.multiThreadedMultiplication(matrix1, matrix2));

        Matrix.stopLibraryWorker();
    }

    @Test
    void multiThreadedAdditionValueTest() throws MatrixDimensionsNotMatchException {
        Matrix m1 = Matrix.createNewFilledMatrix(new double[]{2,-3}, new double[]{-1,4});
        Matrix m2 = Matrix.createNewFilledMatrix(new double[]{2,-3}, new double[]{-1,4});
        System.out.println("Original matrix1:");
        System.out.println(m1);
        System.out.println("Original matrix2:");
        System.out.println(m2);
        System.out.println("Answer:");
        System.out.println(Matrix.multiThreadedAddition(m1,m2));
    }

    @Test
    void multiThreadedAdditionSpeedTest() throws MatrixDimensionsNotMatchException {
        Matrix m1 = Matrix.createNewEmptyMatrix(2000,2000).fillRandomDoubles();
        Matrix m2 = Matrix.createNewEmptyMatrix(2000,2000).fillRandomDoubles();
        long start = System.currentTimeMillis();
        Matrix m3 = Matrix.multiThreadedAddition(m1,m2);
        System.out.println("MT version takes " + (System.currentTimeMillis() - start) + "ms");
        start = System.currentTimeMillis();
        Matrix m5 = Matrix.addition(m1,m2);
        System.out.println("ST version takes " + (System.currentTimeMillis() - start) + "ms");
        assertEquals(m3, m5);
        System.out.println("The result are the same!");
    }


    @Test
    void t(){
        Matrix inputs = Matrix.createNewFilledMatrix(2, 1, randomizeDoubleArray(2));
        System.out.println(inputs);
        inputs.forEach(e -> e+=1);
        System.out.println(inputs);
    }


    DoubleBuffer toDoubleBuffer(double[] doubles){
        DoubleBuffer buffer = BufferUtils.createDoubleBuffer(doubles.length).put(doubles);
        buffer.rewind();
        return buffer;
    }
    void print(DoubleBuffer buffer) {
        for (int i = 0; i < buffer.capacity(); i++) {
            System.out.print(buffer.get(i) + " ");
        }
        System.out.println("");
    }

    @Test
    void cpuMultiplicationTest() throws MatrixDimensionsNotMatchException {
        Matrix matrixA = Matrix.createNewFilledMatrix(new double[]{-1,2,3},new double[]{4,-5,6},new double[]{7,8,-8});
        Matrix matrixB = Matrix.createNewFilledMatrix(new double[]{1,-2,3},new double[]{-4,5,6},new double[]{7,-8,8});
        System.out.println(Matrix.multiplication(matrixA, matrixB));
    }

    @Test
    void openCL() throws IOException {
        Matrix matrixA = Matrix.createNewFilledMatrix(new double[]{1,2,3,4}, new double[]{5,6,7,8}, new double[]{9,10,11,12}, new double[]{13,14,15,16});
        Matrix matrixB = Matrix.createNewFilledMatrix(new double[]{1,2,3,4}, new double[]{5,6,7,8}, new double[]{9,10,11,12}, new double[]{13,14,15,16});
        //Matrix matrixA = Matrix.createNewFilledMatrix(new double[]{1,2}, new double[]{3,4});
        //Matrix matrixB = Matrix.createNewFilledColumnVector(1,2);
        //Matrix matrixA = Matrix.createNewEmptyMatrix(8000,8000).fillRandomDoubles();
        //Matrix matrixB = Matrix.createNewEmptyMatrix(8000,8000).fillRandomDoubles();
        Matrix matrixC = Matrix.createNewEmptyMatrix(matrixA.getNumOfRows(), matrixB.getNumOfColumns());
        final DoubleBuffer a = toDoubleBuffer(matrixA.getData());
        final DoubleBuffer b = toDoubleBuffer(matrixB.getData());
        final DoubleBuffer c = BufferUtils.createDoubleBuffer(matrixA.getNumOfRows() * matrixB.getNumOfColumns());
        final ByteBuffer source = IOUtil.ioResourceToByteBuffer("MatrixMultiplication.cl", 4096);

        try(MemoryStack stack = MemoryStack.stackPush()){
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
            System.out.println("Platform created");

            long device = 0;
            for (int d = 0; d < devices.capacity(); d++) {
                device = devices.get(d);
            }
            System.out.println("Device created");

            long context = clCreateContext(ctxProps, device, CLContextCallback.create(
                    (errinfo, private_info, cb, user_data) -> System.out.println(memUTF8(errinfo))
            ), NULL, errcode_ret);
            InfoUtil.checkCLError(errcode_ret);
            System.out.println("CONTEXT created");
            long queue = clCreateCommandQueue(context, device, NULL, errcode_ret);
            System.out.println("Command Q created");

            long aMemory = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, a, errcode_ret);
            InfoUtil.checkCLError(errcode_ret);
            long bMemory = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, b, errcode_ret);
            InfoUtil.checkCLError(errcode_ret);
            long cMemory = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, c, errcode_ret);
            InfoUtil.checkCLError(errcode_ret);
            clEnqueueWriteBuffer(queue, aMemory, false, 0, a, null, null);
            clEnqueueWriteBuffer(queue, bMemory, false, 0, b, null, null);
            clFinish(queue);
            System.out.println("All Buffer created");


            PointerBuffer strings = BufferUtils.createPointerBuffer(1);
            PointerBuffer lengths = BufferUtils.createPointerBuffer(1);

            strings.put(0, source);
            lengths.put(0, source.remaining());

            long program = clCreateProgramWithSource(context, strings, lengths, errcode_ret);
            checkCLError(errcode_ret);

            // To ensure the program is created before running, a latch is used.
            CountDownLatch latch = new CountDownLatch(1);
            CLProgramCallback buildCallback;
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

            // Time for the real shit
            long kernel = clCreateKernel(program, "multiply", errcode_ret);
            InfoUtil.checkCLError(errcode_ret);
            System.out.println("KERNEL created");

            long start = System.currentTimeMillis();

            // Execution
            PointerBuffer kernel2DGlobalWorkSize = BufferUtils.createPointerBuffer(2);
            PointerBuffer tiles = BufferUtils.createPointerBuffer(2);
            //System.out.println("KERNEL work size created");
            // Set the worksize of A to its size and B to its size <--- WRONG!!!
            // Set the work size of get_global_id(0) to a's num of rows and get_global_id(1) to b's num of columns <--- Correct!!!
            kernel2DGlobalWorkSize.put(0, matrixA.getNumOfRows()).put(1, matrixB.getNumOfColumns());
            tiles.put(0, 4).put(1,4);
            //System.out.println("KERNEL work size copied");
            InfoUtil.checkCLError(errcode_ret);

            clSetKernelArg1i(kernel, 0, matrixA.getNumOfRows());        // set M
            clSetKernelArg1i(kernel, 1, matrixB.getNumOfColumns());     // set N
            clSetKernelArg1i(kernel, 2, matrixA.getNumOfColumns());     // set K
            clSetKernelArg1p(kernel, 3, aMemory);                       // set A    use 1p instead of 1d where p stands for pointer, not flowPoint, as I pass an ARRAY!@!!!!!
            clSetKernelArg1p(kernel, 4, bMemory);                       // set B
            clSetKernelArg1p(kernel, 5, cMemory);                       // set C
            //System.out.println("Args send to kernel");

            clEnqueueNDRangeKernel(queue, kernel, 2, null, kernel2DGlobalWorkSize, tiles, null, null);
            //System.out.println("KERNEL queued created");

            // Read the results memory back into our result buffer
            clEnqueueReadBuffer(queue, cMemory, false, 0, c, null, null);
            //System.out.println("and output ... created");

            clFinish(queue);
            matrixC.setData(c);

            long end = System.currentTimeMillis();

            System.out.println("Time used by OpenCL:" + (end - start) + "ms");

            Matrix matrixD;

            start = System.currentTimeMillis();

            try {
                matrixD = Matrix.multiplication(matrixA, matrixB);
            } catch (MatrixDimensionsNotMatchException e) {
                throw new RuntimeException(e);
            }

            end = System.currentTimeMillis();

            System.out.println("Time used by CPU:" + (end - start) + "ms");

            //assertEquals(matrixD, matrixC);

            // Time to show
            System.out.println(matrixA);
            System.out.println("*");
            System.out.println(matrixB);
            System.out.println("||");
            System.out.println(matrixC);
            System.out.println("vs");
            System.out.println(matrixD);

            //matrixA.toSimpleString();
            //matrixB.toSimpleString();
            //matrixC.toSimpleString();

            // Clean up OpenCL resources
            clReleaseKernel(kernel);
            clReleaseProgram(program);
            clReleaseMemObject(aMemory);
            clReleaseMemObject(bMemory);
            clReleaseMemObject(cMemory);
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
            CL.destroy();
        }


    }

    @Test
    void OpenCLTaskerTest() throws MatrixDimensionsNotMatchException {
        Matrix matrix = Matrix.createNewFilledMatrix(new double[]{2,-3}, new double[]{-1,4});
        Matrix vector = Matrix.createNewEmptyColumnVector(2);

        Matrix.enableOpenCL();

        vector.setColumn(0, new double[]{1, -1});
        System.out.println("Original matrix:");
        System.out.println(matrix);
        System.out.println("Original vector:");
        System.out.println(vector);
        System.out.println("Answer:");
        System.out.println(Matrix.clMultiplication(matrix,vector));

        // ensure thread stop
        Matrix.disableOpenCL();
    }


    @Test
    void NewOpenCLTest(){
        OpenCLInteractor ocli = new OpenCLInteractor();
        ocli.initialize();
        assertThrowsExactly(NullPointerException.class, ()->ocli.exit());   // Expected to throw a NullPointerException
    }
    @Test
    void NewOpenCLTest2(){
        OpenCLInteractor ocli = new OpenCLInteractor();
        ocli.initialize();

        Matrix weights = Matrix.createNewFilledMatrix(2,2, new double[]{1,2,3,4});
        Matrix inputs = Matrix.createNewFilledColumnVector(1,2);
        Matrix bias = Matrix.createNewFilledColumnVector(1,1);
        System.out.println(ocli.clForwardPass(weights,inputs,bias, 0));
        ocli.exit();

    }



}
