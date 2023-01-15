package objects;

public record MatrixDimension(int numOfRows, int numOfCols) {
    @Override
    public boolean equals(Object obj) {
        if(obj.getClass() != MatrixDimension.class)
            return false;
        MatrixDimension matrixDimension = (MatrixDimension) obj;
        if(this.numOfRows != matrixDimension.numOfRows || this.numOfCols != matrixDimension.numOfCols)
            return false;
        return true;
    }
}
