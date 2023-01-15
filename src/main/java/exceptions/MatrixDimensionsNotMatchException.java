package exceptions;

public class MatrixDimensionsNotMatchException extends Exception {
    public MatrixDimensionsNotMatchException() {
    }

    public MatrixDimensionsNotMatchException(String message) {
        super(message);
    }

    public MatrixDimensionsNotMatchException(String message, Throwable cause) {
        super(message, cause);
    }

    public MatrixDimensionsNotMatchException(Throwable cause) {
        super(cause);
    }

    public MatrixDimensionsNotMatchException(String message, Throwable cause, boolean enableSuppression, boolean writableStackTrace) {
        super(message, cause, enableSuppression, writableStackTrace);
    }
}
