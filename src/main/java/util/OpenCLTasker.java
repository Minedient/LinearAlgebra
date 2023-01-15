package util;

import exceptions.MatrixDimensionsNotMatchException;
import objects.Matrix;
import objects.TwoMatrix;

public final class OpenCLTasker {

    OpenCLWorker[] workers = new OpenCLWorker[1];

    public OpenCLTasker() {
        this.workers[0] = new OpenCLWorker();
    }

    public synchronized Matrix clMatrixMultiply(Matrix matrixA, Matrix matrixB) throws MatrixDimensionsNotMatchException {
        if (matrixA.getNumOfColumns() != matrixB.getNumOfRows())
            throw new MatrixDimensionsNotMatchException();
        Matrix result;
        this.workers[0].sendTask(new TwoMatrix(matrixA, matrixB));
        while ((result = workers[0].resultQueue.poll()) == null) {

        }
        return result;
    }

    public void stopTask() {
        this.workers[0].interrupt();
    }

    public void startTask() {
        this.workers[0].start();
    }
}
