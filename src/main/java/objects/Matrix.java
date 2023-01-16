package objects;

import exceptions.MatrixDimensionsNotMatchException;
import exceptions.MatrixIndexOutofBoundException;
import exceptions.MatrixInitialSizeException;
import exceptions.NoInverseException;
import util.OpenCLTasker;

import java.nio.DoubleBuffer;
import java.util.Objects;
import java.util.concurrent.*;
import java.util.function.DoubleFunction;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

/**
 * Matrix class
 *
 * @author Minedient
 */
public class Matrix {

    public static final int NUM_OF_THREADS = Runtime.getRuntime().availableProcessors();
    private static final OpenCLTasker clTasker = new OpenCLTasker();
    /**
     * The monitor for all worker threads, the number of worker
     * threads spawned is equals to the computer's available cores
     */
    private static final ParallelMatrixMonitor pm = new ParallelMatrixMonitor(NUM_OF_THREADS);
    private final double[] data;
    private final int numOfRows;    // AKA columnSize
    private final int numOfColumns; // AKA rowSize

    /**
     * Create a new {@code Matrix} given its number of rows and columns
     * @param numOfRows     Matrix's number of rows
     * @param numOfColumns  Matrix's number of columns
     */
    Matrix(int numOfRows, int numOfColumns) {
        this.numOfRows = numOfRows;
        this.numOfColumns = numOfColumns;
        data = new double[numOfRows * numOfColumns];
    }

    /**
     * Create a new Matrix using {@code MatrixDimension}
     * @param dimension The dimension of the {@code Matrix}
     */
    private Matrix(MatrixDimension dimension) {
        this(dimension.numOfRows(), dimension.numOfCols());
    }

    private Matrix(MatrixDimension dimension, INIT_MODE mode) {
        this(dimension.numOfRows(), dimension.numOfCols());
        if (Objects.requireNonNull(mode) == INIT_MODE.IDENTITY) {
            initializeAsIdentity();
        }
    }

    private Matrix(Matrix matrix) {
        this(matrix.getDimension());
        System.arraycopy(matrix.data, 0, this.data, 0, matrix.getNumOfEntries());
    }

    private Matrix(double[]... data) {
        int rowSize = data.length;
        for (int i = 1; i < data.length; i++)
            if (data[1].length != rowSize)
                throw new MatrixInitialSizeException();
        this.numOfRows = rowSize;
        this.numOfColumns = data[0].length;
        this.data = Stream.of(data).flatMapToDouble(DoubleStream::of).toArray();
    }

    /**
     * Create an empty {@code Matrix} given the number of rows and columns
     * @param numOfRows     Matrix's number of rows
     * @param numOfColumns  Matrix's number of columns
     * @return An empty matrix with given number of rows and columns
     */
    public static Matrix createNewEmptyMatrix(int numOfRows, int numOfColumns) {
        return new Matrix(numOfRows, numOfColumns);
    }

    /**
     * Create a new matrix given an array of double arrays.
     * <p>
     *     {@code Matrix.createNewFilledMatrix(new double[]{1,2,3}, new double[]{4,5,6};}
     * </p>
     *
     * Noticed that if the length of each subarray doesn't match the first subarray,
     * a {@code MatrixInitialSizeException} with be thrown
     *
     * @param data  The array of double arrays.
     * @return A new matrix with given data
     */
    public static Matrix createNewFilledMatrix(double[]... data) {
        return new Matrix(data);
    }

    /**
     * Create a new matrix given its number of rows and columns and an array of doubles
     *
     * Noticed that if the length of the array doesn't match the product of rows and columns
     * of the Matrix, a {@code MatrixInitialSizeException} with be thrown
     *
     * @param numOfRows     Matrix's number of rows
     * @param numOfColumns  Matrix's number of columns
     * @param data          Matrix's data
     * @return A new matrix with given rows, columns and data.
     */
    public static Matrix createNewFilledMatrix(int numOfRows, int numOfColumns, double... data) {
        if (numOfRows * numOfColumns != data.length)
            throw new MatrixInitialSizeException();
        Matrix result = new Matrix(numOfRows, numOfColumns);
        System.arraycopy(data, 0, result.data, 0, data.length);
        return result;
    }

    /**
     * Create an empty {@code Matrix} given a {@code MatrixDimension}
     * @param dimension     The dimension of the matrix
     * @return  An empty matrix with given dimension
     */
    public static Matrix createNewEmptyMatrix(MatrixDimension dimension) {
        return new Matrix(dimension);
    }

    /**
     * Create an empty {@code Matrix} that represent a Row Vector
     * @param size      The size of the Row Vector
     * @return  An empty matrix with dimension [1, n] where n is the size
     */
    public static Matrix createNewEmptyRowVector(int size) {
        return new Matrix(1, size);
    }

    /**
     * Create a filled Row Vector with given data
     * @param data      The data
     * @return  A filled matrix with dimension [1, n] where n is the size
     */
    public static Matrix createNewFilledRowVector(double... data) {
        Matrix rowVector = createNewFilledMatrix(1, data.length, data);
        return rowVector;
    }

    /**
     * Create an empty {@code Matrix} that represent a Column Vector
     * @param size      The size of the Column Vector
     * @return  An empty matrix with dimension [n, 1] where n is the size
     */
    public static Matrix createNewEmptyColumnVector(int size) {
        return new Matrix(size, 1);
    }

    /**
     * Create a filled Column Vector with given data
     * @param data      The data
     * @return  A filled matrix with dimension [n, 1] where n is the size
     */
    public static Matrix createNewFilledColumnVector(double... data) {
        Matrix columnVector = createNewFilledMatrix(data.length, 1, data);
        return columnVector;
    }

    /**
     * Create a new instance of {@code Matrix} that resemble the original Matrix
     * @param matrix    The matrix to copy from
     * @return  A new instance of {@code Matrix} that resemble the original Matrix.
     */
    public static Matrix copyingMatrix(Matrix matrix) {
        return new Matrix(matrix);
    }

    /**
     * Check if two {@code Matrix} have the same size (dimension).
     *
     * @param m1    The first matrix
     * @param m2    The second matrix
     * @return {@code MatrixDimension}, if they are the same size
     * @throws MatrixDimensionsNotMatchException
     */
    public static MatrixDimension ensureSameSize(Matrix m1, Matrix m2) throws MatrixDimensionsNotMatchException {
        if (m1.numOfRows != m2.numOfRows || m1.numOfColumns != m2.numOfColumns)
            throw new MatrixDimensionsNotMatchException("The dimension of Matrix mismatch");
        return new MatrixDimension(m1.numOfRows, m2.numOfColumns);
    }

    /**
     * Add two matrix
     *
     * @param m1    The first matrix
     * @param m2    The second matrix
     * @return answer
     * @throws MatrixDimensionsNotMatchException
     */
    public static Matrix addition(Matrix m1, Matrix m2) throws MatrixDimensionsNotMatchException {
        Matrix result = new Matrix(ensureSameSize(m1, m2));
        for (int i = 0; i < result.getNumOfEntries(); i++) {
            result.data[i] = m1.data[i] + m2.data[i];
        }
        return result;
    }

    /**
     * Subtract to matrix
     *
     * @param m1    The first matrix
     * @param m2    The second matrix
     * @return answer
     * @throws MatrixDimensionsNotMatchException
     */
    public static Matrix subtraction(Matrix m1, Matrix m2) throws MatrixDimensionsNotMatchException {
        Matrix result = new Matrix(ensureSameSize(m1, m2));
        for (int i = 0; i < result.getNumOfEntries(); i++) {
            result.data[i] = m1.data[i] - m2.data[i];
        }
        return result;
    }

    /**
     * Do a scalar multiplication on matrix
     * @param matrix    The matrix to be multiplied
     * @param scale     The scale
     * @return  The scaled matrix
     */
    public static Matrix scalarMultiplication(Matrix matrix, double scale) {
        Matrix result = new Matrix(matrix);
        for (int i = 0; i < result.getNumOfEntries(); i++)
            result.data[i] *= scale;
        return result;
    }

    /**
     * Transpose matrix
     * @param matrix   The matrix to transpose
     * @return  The transposed matrix
     */
    public static Matrix transpose(Matrix matrix) {
        Matrix result = new Matrix(matrix.numOfColumns, matrix.numOfRows);
        for (int i = 0; i < matrix.numOfRows; i++)
            for (int j = 0; j < matrix.numOfColumns; j++)
                result.setDatum(j, i, matrix.getDatum(i, j));
        return result;
    }

    /**
     * Find the first non-zero entry in the array that its index is larger than the limiter,
     * If no matching value, it will return -1
     *
     * @param subArray The subArray in the matrix
     * @param limiter  The limiting index
     * @return The first non-zero data's index, otherwise -1
     */
    private static int firstNonZeroWithLimit(double[] subArray, int limiter) {
        for (int i = limiter; i < subArray.length; i++)
            if (subArray[i] != 0) // if the value at index[i-1] is not zero
                return i;
        return -1;
    }

    /**
     * Perform LU Decomposition on the given matrix
     * @param matrix    The matrix to be de-composited.
     * @return  The de-composited matrix
     */
    public static LUMatrixGroup luDecomposition(Matrix matrix) {
        Matrix upper = new Matrix(matrix);
        Matrix lower = new Matrix(matrix.getDimension(), INIT_MODE.IDENTITY);
        Matrix permutation = new Matrix(matrix.getDimension(), INIT_MODE.IDENTITY);
        for (int pivotRow = 0; pivotRow < upper.numOfRows; pivotRow++) {    // Last row don't have to perform Gaussian elimination
            double[] column = upper.getColumn(pivotRow);
            int firstNonZero = firstNonZeroWithLimit(column, pivotRow);  // Find the first non-zero element
            if (firstNonZero != -1) {                                    // If there is a left-most non zero element
                if (pivotRow != firstNonZero) { // Make the marked row as the pivot row by swapping
                    upper.swapRows(pivotRow, firstNonZero);
                    permutation.swapRows(pivotRow, firstNonZero);
                }
                for (int movingRow = pivotRow + 1; movingRow < upper.numOfRows; movingRow++) {
                    double lFactor = upper.getRow(movingRow)[pivotRow] / upper.getRow(firstNonZero)[pivotRow];
                    lower.setDatum(firstNonZero, movingRow, lFactor);
                    upper.addMultiplesToRow(firstNonZero, movingRow, -lFactor);
                }
            }
        }
        lower = Matrix.transpose(lower);    // Transpose it to get the final lower matrix
        return new LUMatrixGroup(permutation, lower, upper);
    }

    /**
     * Inverse the given matrix
     *
     * @param matrix The original
     * @return  The inverse matrix of the original matrix
     */
    public static Matrix inverse(Matrix matrix) {
        Matrix cloned = new Matrix(matrix);
        Matrix result = new Matrix(matrix.getDimension(), INIT_MODE.IDENTITY);
        // Working down from the top
        for (int pivotRow = 0; pivotRow < cloned.getNumOfRows(); pivotRow++) { // Last row don't have to perform Gaussian elimination
            double[] column = cloned.getColumn(pivotRow);
            int firstNonZero = firstNonZeroWithLimit(column, pivotRow);  // Find the first non-zero element
            if (firstNonZero != -1) {                                    // If there is a left-most non zero element
                if (pivotRow != firstNonZero) { // Make the marked row as the pivot row by swapping
                    cloned.swapRows(pivotRow, firstNonZero);
                    result.swapRows(pivotRow, firstNonZero);
                }
                for (int movingRow = pivotRow + 1; movingRow < cloned.getNumOfRows(); movingRow++) {
                    findMultipleAndEliminate(cloned, result, pivotRow, movingRow);
                }
            }
        }
        //Working up from the bottom
        for (int pivotRow = cloned.getNumOfRows() - 1; pivotRow >= 0; pivotRow--) {
            for (int movingRow = pivotRow - 1; movingRow >= 0; movingRow--) {
                findMultipleAndEliminate(cloned, result, pivotRow, movingRow);
            }
        }
        //Turn all into 1s
        for (int pivotRow = 0; pivotRow < cloned.getNumOfRows(); pivotRow++) {
            double datum = cloned.getDatum(pivotRow, pivotRow);
            if (datum != 0) {
                cloned.multiplyRow(pivotRow, 1 / datum);
                result.multiplyRow(pivotRow, 1 / datum);
            }
        }
        return result;
    }

    /**
     * Find the multiple of the source row needed to add/subtract in order to reduce the target zero non-diagonal number to zero
     *
     * @param original  The original matrix to be reduced to R.E.F/R.E.E.F
     * @param augment   The augment matrix to store the result
     * @param sourceRow The source row
     * @param targetRow The target row
     * @throws NoInverseException Throw if the program encounter problems
     */
    private static void findMultipleAndEliminate(Matrix original, Matrix augment, int sourceRow, int targetRow) {
        double factor = original.getRow(targetRow)[sourceRow];
        if (factor == 0.0)
            return;
        try {
            double lFactor = factor / original.getRow(sourceRow)[sourceRow];    //throw exception if divider is 0(don't have inverse)
            original.addMultiplesToRow(sourceRow, targetRow, -lFactor);
            augment.addMultiplesToRow(sourceRow, targetRow, -lFactor);
        } catch (ArithmeticException arithmeticException) {
            throw new NoInverseException("This matrix can not be inverted?");
        }
    }

    /**
     * Multiply two matrix together and return a new Matrix.
     * The matrix must meet the requirement for multiplying Matrix.
     *
     * @param matrixA The first matrix
     * @param matrixB The second matrix
     * @return The result of the multiplication
     * @throws MatrixDimensionsNotMatchException
     */
    public static Matrix multiplication(Matrix matrixA, Matrix matrixB) throws MatrixDimensionsNotMatchException {
        if (matrixA.numOfColumns != matrixB.numOfRows)
            throw new MatrixDimensionsNotMatchException();
        Matrix result = new Matrix(matrixA.numOfRows, matrixB.numOfColumns);
        for (int row = 0; row < result.numOfRows; row++) {
            for (int column = 0; column < result.numOfColumns; column++) {
                double sum = 0;
                double[] aRow = matrixA.getRow(row);
                double[] bColumn = matrixB.getColumn(column);
                for (int times = 0; times < matrixA.numOfColumns; times++)
                    sum += aRow[times] * bColumn[times];
                result.setDatum(row, column, sum);
            }
        }
        return result;
    }

    /**
     * Find the determinant of the matrix (Square matrix only)
     *
     * @param matrix THe matrix to find its determinant
     * @return The determinant of the square matrix
     * @throws MatrixDimensionsNotMatchException if it is not a square matrix
     */
    public static double determinant(Matrix matrix) throws MatrixDimensionsNotMatchException {
        if (!matrix.isSquareMatrix())
            throw new MatrixDimensionsNotMatchException("A square matrix is required!");
        Matrix u = Matrix.luDecomposition(matrix).u();
        double determinant = 1.0;
        for (int i = 0; i < u.numOfRows; i++)
            determinant *= u.getDatum(i, i);
        return determinant;
    }

    /**
     * Create a square matrix initialized as identity matrix
     * @param size  The size of the matrix [size, size]
     * @return  The identity matrix.
     */
    public static Matrix createIdentity(int size) {
        Matrix x = createNewEmptyMatrix(size, size);
        for (int i = 0; i < x.numOfRows; i++)
            x.setDatum(i, i, 1);
        return x;
    }

    /**
     * Multiply two matrix together and return a new Matrix.
     * The matrix must meet the requirement for multiplying Matrix.
     * <p>
     * This version use OpenCL implementation.
     * Noticed that for smaller matrix, the single threaded cpu version is generally faster in execution
     * @see Matrix#multiplication(Matrix, Matrix)
     *
     * @param matrixA The first matrix
     * @param matrixB The second matrix
     * @return The result of the multiplication
     * @throws MatrixDimensionsNotMatchException
     */
    public static Matrix clMultiplication(Matrix matrixA, Matrix matrixB) throws MatrixDimensionsNotMatchException {
        return clTasker.clMatrixMultiply(matrixA, matrixB);
    }

    /**
     * Multiply two matrix together and return a new Matrix.
     * The matrix must meet the requirement for multiplying Matrix.
     * <br>
     * This version is multiThreaded.
     * Noticed that for smaller matrix, the single threaded cpu version is generally faster in execution
     * @see Matrix#multiplication(Matrix, Matrix)
     *
     * @param matrixA The first matrix
     * @param matrixB The second matrix
     * @return The result of the multiplication
     * @throws MatrixDimensionsNotMatchException
     */
    public static Matrix multiThreadedMultiplication(Matrix matrixA, Matrix matrixB) throws MatrixDimensionsNotMatchException {
        if (matrixA.getNumOfColumns() != matrixB.getNumOfRows())
            throw new MatrixDimensionsNotMatchException();
        Matrix result = Matrix.createNewEmptyMatrix(matrixA.getNumOfRows(), matrixB.getNumOfColumns());
        // Use this to ensure Matrix multiply correctly
        CountDownLatch latch = new CountDownLatch(result.numOfRows);
        for (int i = 0; i < result.getNumOfRows(); i++) {
            pm.giveRequest(new MultiplyRequest(result, matrixA, matrixB, i, latch));
        }
        try {
            latch.await();
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
        return result;
    }

    /**
     * Add two matrix
     * <br>
     * This version is multiThreaded.
     * Noticed that for smaller matrix, the single threaded cpu version is generally faster in execution
     *
     * @see Matrix#addition(Matrix, Matrix)
     *
     * @param m1    The first matrix
     * @param m2    The second matrix
     * @return answer
     * @throws MatrixDimensionsNotMatchException
     */
    public static Matrix multiThreadedAddition(Matrix m1, Matrix m2) throws MatrixDimensionsNotMatchException{
        Matrix result = new Matrix(ensureSameSize(m1, m2));
        CountDownLatch latch = new CountDownLatch(result.numOfRows);
        for (int i=0; i<result.numOfRows;i++){
            pm.giveRequest(new AdditionRequest(result, m1, m2, i , latch));
        }
        try {
            latch.await();
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
        return result;
    }

    public static void startOfLibrary() {
        clTasker.startTask();
    }

    public static void endOfLibrary() {
        clTasker.stopTask();
    }

    public static void stopLibraryWorker() {
        pm.stopWorkers();
    }

    /**
     * Get the number of rows of this Matrix
     * @return  The number of rows
     */
    public int getNumOfRows() {
        return numOfRows;
    }
    /**
     * Get the number of columns of this Matrix
     * @return  The number of columns
     */
    public int getNumOfColumns() {
        return numOfColumns;
    }

    /**
     * Get the size of a single row
     * In the implementation it is equal to getNumOfColumns() as
     * they have the same meaning.
     *
     * @see Matrix#getNumOfColumns()
     *
     * @return  The size of a single row
     */
    public int getRowSize() {
        return numOfColumns;
    }

    /**
     * Get the size of a single column
     * In the implementation it is equal to getNumOfRows() as
     * they have the same meaning.
     *
     * @see Matrix#getNumOfRows()
     *
     * @return  The size of a single column
     */
    public int getColumnSize() {
        return numOfRows;
    }

    /**
     * Get the number of entries (data) in this matrix
     * @return The number of entries
     */
    public int getNumOfEntries() {
        return data.length;
    }

    /**
     * Get a single datum in this matrix
     * @param rowNumber     The row number of the datum
     * @param columnNumber  The column number of the datum
     * @return The datum
     */
    public double getDatum(int rowNumber, int columnNumber) {
        if (rowNumber > numOfRows || rowNumber < 0 || columnNumber > numOfColumns || columnNumber < 0)
            throw new MatrixIndexOutofBoundException();
        return this.data[columnNumber + rowNumber * getRowSize()];
    }

    /**
     * Get a row from this matrix
     * @param rowNumber     The row number
     * @return  The row
     */
    public double[] getRow(int rowNumber) {
        if (rowNumber > numOfRows || rowNumber < 0)
            throw new MatrixIndexOutofBoundException();
        double[] result = new double[getRowSize()];
        System.arraycopy(this.data, rowNumber * getRowSize(), result, 0, getRowSize());
        return result;
    }

    /**
     * Get a column from this matrix
     * @param columnNumber  The column number
     * @return  The column
     */
    public double[] getColumn(int columnNumber) {
        if (columnNumber > numOfColumns || columnNumber < 0)
            throw new MatrixIndexOutofBoundException();
        double[] result = new double[getColumnSize()];
        for (int i = 0; i < getColumnSize(); i++)
            result[i] = this.data[i * getRowSize() + columnNumber];
        return result;
    }

    /**
     * Set a single datum in this matrix
     * @param rowNumber     The row number of the datum
     * @param columnNumber  The column number of the datum
     * @param datum         The datum to set
     */
    public void setDatum(int rowNumber, int columnNumber, double datum) {
        if (rowNumber > numOfRows || rowNumber < 0 || columnNumber > numOfColumns || columnNumber < 0)
            throw new MatrixIndexOutofBoundException();
        this.data[columnNumber + rowNumber * numOfColumns] = datum;
    }

    /**
     * Set a single row in this matrix
     * @param rowNumber     The row number of the row to set
     * @param newData       The data to set
     */
    public void setRow(int rowNumber, double[] newData) {
        if (rowNumber > numOfRows || rowNumber < 0 || newData.length > getRowSize())
            throw new MatrixIndexOutofBoundException();
        System.arraycopy(newData, 0, this.data, rowNumber * getRowSize(), getRowSize());
    }

    /**
     * Set a single column in this matrix
     * @param columnNumber  The column number of the column to set
     * @param newData       The data to set
     */
    public void setColumn(int columnNumber, double[] newData) {
        if (columnNumber > numOfColumns || columnNumber < 0 || newData.length > getColumnSize())
            throw new MatrixIndexOutofBoundException();
        for (int i = 0; i < getColumnSize(); i++)
            this.data[i * getRowSize() + columnNumber] = newData[i];
    }

    /**
     * Swap rows in this Matrix
     *
     * @param sourceRow The source row
     * @param targetRow The target row to swap
     */
    public void swapRows(int sourceRow, int targetRow) {
        double[] tempRow = getRow(sourceRow);
        setRow(sourceRow, getRow(targetRow));
        setRow(targetRow, tempRow);
    }

    /**
     * Swap columns in this Matrix
     *
     * @param sourceColumn The source column
     * @param targetColumn The target column to swap
     */
    public void swapColumn(int sourceColumn, int targetColumn) {
        double[] tempColumn = getColumn(sourceColumn);
        setColumn(sourceColumn, getColumn(targetColumn));
        setColumn(targetColumn, tempColumn);
    }

    /**
     * Multiply a row in this Matrix
     * @param rowNumber     The row to be multiplied
     * @param scale         The scale factor of the row
     */
    public void multiplyRow(int rowNumber, double scale) {
        double[] tempRow = getRow(rowNumber);
        for (int i = 0; i < tempRow.length; i++)
            tempRow[i] *= scale;
        setRow(rowNumber, tempRow);
    }

    /**
     * Multiply a column in this Matrix
     * @param columnNumber  The column to be multiplied
     * @param scale         The scale factor of the row
     */
    public void multiplyColumn(int columnNumber, double scale) {
        double[] tempColumn = getColumn(columnNumber);
        for (int i = 0; i < tempColumn.length; i++)
            tempColumn[i] *= scale;
        setColumn(columnNumber, tempColumn);
    }

    /**
     * Add multiples of Row A to Row B in this Matrix
     * @param sourceRow     The source row (rowA)
     * @param targetRow     The target row (rowB)
     * @param scale         The scale of source row
     */
    public void addMultiplesToRow(int sourceRow, int targetRow, double scale) {
        double[] tempRow = getRow(sourceRow);
        double[] newValues = getRow(targetRow);
        for (int i = 0; i < tempRow.length; i++)
            newValues[i] += (tempRow[i] * scale);
        setRow(targetRow, newValues);
    }


    /**
     * Add multiples of Column A to Column B in this Matrix
     * @param sourceColumn     The source column (columnA)
     * @param targetColumn     The target column (columnB)
     * @param scale         The scale of source column
     */
    public void addMultiplesToColumn(int sourceColumn, int targetColumn, double scale) {
        double[] tempColumn = getColumn(sourceColumn);
        double[] newValues = getColumn(targetColumn);
        for (int i = 0; i < tempColumn.length; i++)
            newValues[i] += (tempColumn[i] * scale);
        setColumn(targetColumn, newValues);
    }

    /**
     * Multiply each entry in the Matrix m to each entry this Matrix (In-place)
     *
     * @param m The matrix to multiply
     * @return  The result matrix
     * @throws MatrixDimensionsNotMatchException
     */
    public Matrix linearMultiplication(Matrix m) throws MatrixDimensionsNotMatchException {
        Matrix.ensureSameSize(this, m);
        for (int i = 0; i < this.data.length; i++) {
            this.data[i] *= m.data[i];
        }
        return this;
    }

    /**
     * Initialize this matrix as an Identity matrix
     * <br>
     * Notice that it does not perform check on its dimension
     */
    private void initializeAsIdentity() {
        for (int i = 0; i < this.numOfRows; i++)
            setDatum(i, i, 1);
    }

    /**
     * Check if this matrix is a square matrix
     * @return true if it is a square matrix, otherwise false
     */
    public boolean isSquareMatrix() {
        return this.numOfRows == this.numOfColumns;
    }

    /**
     * Get a {@code MatrixDimension} object of this matrix
     * @return  The object
     */
    public MatrixDimension getDimension() {
        return new MatrixDimension(numOfRows, numOfColumns);
    }

    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        for (int i = 0; i < this.numOfRows; i++) {
            stringBuilder.append("|");
            for (int j = 0; j < this.numOfColumns; j++) {
                stringBuilder.append(getDatum(i, j));
                if (j != this.numOfColumns - 1)
                    stringBuilder.append(" ");
            }
            stringBuilder.append("|");
            if (i != this.numOfRows - 1)
                stringBuilder.append("\n");
        }
        return stringBuilder.toString();
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == null)
            return false;
        if (obj.getClass() != Matrix.class)
            return false;
        Matrix newMatrix = (Matrix) obj;
        if (!newMatrix.getDimension().equals(this.getDimension()))
            return false;
        for (int i = 0; i < getNumOfEntries(); i++)
            if (this.data[i] != newMatrix.data[i])
                return false;
        return true;

    }

    /**
     * Fill the matrix with random doubles
     * @return this matrix
     */
    public Matrix fillRandomDoubles() {
        for (int i = 0; i < data.length; i++) {
            data[i] = Math.random();
        }
        return this;
    }

    /**
     * Perform functions on all entries in this matrix (in-place)
     * <br>
     * Example: {@code matrix.forEach(e => e+1);}
     *
     * @param function  The function to perform
     */
    public void forEach(DoubleFunction<Double> function) {
        for (int i = 0; i < data.length; i++)
            data[i] = function.apply(data[i]);
    }

    /**
     * Get the data array of this matrix
     * @return  The data array
     */
    public double[] getData() {
        return this.data;
    }

    /**
     * Set the data of this matrix given a {@code DoubleBuffer} (used in OpenCL matrix function).
     * @param buffer The DoubleBuffer object
     */
    public void setData(DoubleBuffer buffer) {
        double[] arr = new double[buffer.capacity()];
        for (int i = 0; i < buffer.capacity(); i++) {
            arr[i] = buffer.get(i);
        }
        System.arraycopy(arr, 0, this.data, 0, buffer.capacity());
    }

    /**
     * Get the data array of this matrix as a float array
     * @return  The data array
     */
    public float[] getDataAsFloat() {
        float[] fs = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            fs[i] = (float) data[i];
        }
        return fs;
    }

    /**
     * Print the matrix without newline
     */
    public void toSimpleString() {
        System.out.print("[");
        for (int i = 0; i < data.length; i++) {
            System.out.print(data[i]);
            if (i != data.length - 1)
                System.out.print(", ");
        }
        System.out.print("]\n");
    }

    /**
     * Initialize mode of the matrix
     */
    private enum INIT_MODE {NULL, IDENTITY}
}

/**
 * A work assignment class aimed to provide work for {@code RowWorker}.
 */
class ParallelMatrixMonitor {

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

    public void stopWorkers() {
        for (int i = 0; i < threadPool.length; i++) {
            threadPool[i].terminate();
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

    public synchronized CalculationRequest takeRequest() {
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
        return request;
    }
}

/**
 * Get request from {@code ParallelMatrixMonitor} and run calculations when it is alive.
 * <br>
 * Stop processing any request after it has been terminated.
 */
class Worker extends Thread {
    private final ParallelMatrixMonitor monitor;
    private boolean alive = true;

    Worker(ParallelMatrixMonitor monitor) {
        this.monitor = monitor;
    }

    public void terminate() {
        alive = false;
    }

    @Override
    public void run() {
        while (alive) {
            CalculationRequest request = monitor.takeRequest();
            request.calculate();
        }
    }


}

/**
 * A Request that specify the Matrix to be multiplied and the row of multiplication.
 * <p>
 * The request is created by {@code Matrix} and consumed by {@code RowMultiplyWorker}.
 * <p>
 * In order to ensure the whole matrix is calculated before continue in the main thread,
 * a {@code CountDownLatch} is used.
 * </p>
 */
class MultiplyRequest extends BaseRequest{
    public MultiplyRequest(Matrix container, Matrix matrixA, Matrix matrixB, int row, CountDownLatch latch) {
        super(container, matrixA, matrixB, row, latch);
    }

    @Override
    public void calculate() {
        double answer;
        for (int i = 0; i < matrixB.getNumOfColumns(); i++) {
            answer = 0.0;
            for (int j = 0; j < matrixA.getNumOfColumns(); j++) {
                answer += matrixA.getDatum(row, j) * matrixB.getDatum(j, i);
            }
            container.setDatum(row, i, answer);
        }
        latch.countDown();
    }
}

/**
 * A Request that specify the Matrix to be added and the row of additional.
 * <p>
 * The request is created by {@code Matrix} and consumed by {@code RowMultiplyWorker}.
 * <p>
 * In order to ensure the whole matrix is calculated before continue in the main thread,
 * a {@code CountDownLatch} is used.
 * </p>
 */
class AdditionRequest extends BaseRequest {
    public AdditionRequest(Matrix container, Matrix matrixA, Matrix matrixB, int row, CountDownLatch latch){
        super(container, matrixA, matrixB, row, latch);
    }

    @Override
    public void calculate() {
        for (int i = 0; i < matrixA.getNumOfColumns(); i++) {
            container.setDatum(row, i, matrixA.getDatum(row, i) + matrixB.getDatum(row, i));
        }
        latch.countDown();
    }
}

