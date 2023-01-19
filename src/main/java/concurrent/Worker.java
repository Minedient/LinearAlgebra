package concurrent;

import objects.CalculationRequest;

import java.util.Optional;

/**
 * Get request from {@code ParallelMatrixMonitor} and run calculations when it is alive.
 * <br>
 * Stop processing any request after it has been terminated.
 */
public class Worker extends Thread {
    private final ParallelMatrixMonitor monitor;

    private volatile boolean alive = true;

    Worker(ParallelMatrixMonitor monitor) {
        this.monitor = monitor;
    }

    public void terminate() {
        alive = false;
    }

    @Override
    public void run() {
        while (alive) {
            Optional<CalculationRequest> request = monitor.takeRequest();
            request.ifPresent(CalculationRequest::calculate);
        }
    }


}
