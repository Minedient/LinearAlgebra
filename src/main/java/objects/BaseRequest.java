package objects;

import java.util.concurrent.CountDownLatch;

public abstract class BaseRequest implements CalculationRequest {
    protected final Matrix container;
    protected final Matrix matrixA;
    protected final Matrix matrixB;
    protected final int row;
    protected final CountDownLatch latch;

    public BaseRequest(Matrix container, Matrix matrixA, Matrix matrixB, int row, CountDownLatch latch) {
        this.container = container;
        this.matrixA = matrixA;
        this.matrixB = matrixB;
        this.row = row;
        this.latch = latch;
    }
}
