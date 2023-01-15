package exceptions;

public class MatrixInitialSizeException extends ArrayIndexOutOfBoundsException{
    public MatrixInitialSizeException() {
        super();
    }

    public MatrixInitialSizeException(String s) {
        super(s);
    }

    public MatrixInitialSizeException(int index) {
        super(index);
    }
}
