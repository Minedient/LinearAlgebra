package exceptions;

/**
 * Thrown to indicate that a matrix has been accessed with an illegal index. The index is either negative or greater than or equal to the dimension of the matrix.
 * @author Minedient
 */
public class MatrixIndexOutofBoundException extends ArrayIndexOutOfBoundsException{
    public MatrixIndexOutofBoundException() {
    }

    public MatrixIndexOutofBoundException(String s) {
        super(s);
    }

    public MatrixIndexOutofBoundException(int index) {
        super(index);
    }
}
