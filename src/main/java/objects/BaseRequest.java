package objects;

import java.util.concurrent.CountDownLatch;

/**
 * The base class for every {@code Request}.
 * It contains necessary information for {@code Worker} to
 * perform calculation on the container
 */
public abstract class BaseRequest implements CalculationRequest {
    /**
     * Container Matrix that contain the result
     */
    protected final Matrix container;
    /**
     * First Matrix
     */
    protected final Matrix matrixA;
    /**
     * Second Matrix
     */
    protected final Matrix matrixB;
    /**
     * Row this Request designated for
     */
    protected final int row;
    /**
     * Latch that used to halt the main thread until all worker finish the work
     * in order to ensure data integrity
     */
    protected final CountDownLatch latch;

    public BaseRequest(Matrix container, Matrix matrixA, Matrix matrixB, int row, CountDownLatch latch) {
        this.container = container;
        this.matrixA = matrixA;
        this.matrixB = matrixB;
        this.row = row;
        this.latch = latch;
    }
}

abstract class MultiRowRequest extends BaseRequest{

    protected final int toRow;

    public MultiRowRequest(Matrix container, Matrix matrixA, Matrix matrixB, int fromRow, int toRow, CountDownLatch latch) {
        super(container, matrixA, matrixB, fromRow, latch);
        this.toRow = toRow;
    }
}