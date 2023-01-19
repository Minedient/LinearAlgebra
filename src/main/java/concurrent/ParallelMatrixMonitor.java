package concurrent;

import objects.CalculationRequest;

import java.util.Optional;

/**
 * A work assignment class aimed to provide work for {@code RowWorker}.
 */
public class ParallelMatrixMonitor {

    private static final int MAX_REQUEST = 128;
    private final Worker[] threadPool;
    private final CalculationRequest[] requestsQueue;
    private int tail;
    private int head;
    private int count;

    public ParallelMatrixMonitor(int numOfThreads) {
        this.requestsQueue = new CalculationRequest[MAX_REQUEST];
        this.head = 0;
        this.tail = 0;
        this.count = 0;
        this.threadPool = new Worker[numOfThreads];
        for (int i = 0; i < numOfThreads; i++) {
            threadPool[i] = new Worker(this);
        }
        startWorkers();
    }

    public void startWorkers() {
        for (int i = 0; i < threadPool.length; i++) {
            threadPool[i].start();
        }
    }

    public synchronized void stopWorkers() {
        for (int i = 0; i < threadPool.length; i++) {
            threadPool[i].terminate();
            // Dummies request equals to the size of thread pool is created in order to
            // bring all threads out of the wait() method.
            giveRequest(() -> {
            });
        }
    }

    public synchronized void giveRequest(CalculationRequest request) {
        while (count >= requestsQueue.length) {
            try {
                wait();
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }
        requestsQueue[tail] = request;
        tail = (tail + 1) % requestsQueue.length;
        count++;
        notifyAll();
    }

    public synchronized Optional<CalculationRequest> takeRequest() {
        while (count <= 0) {
            try {
                wait();
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }
        CalculationRequest request = requestsQueue[head];
        head = (head + 1) % requestsQueue.length;
        count--;
        notifyAll();
        return Optional.ofNullable(request);
    }
}
