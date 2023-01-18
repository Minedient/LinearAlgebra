package objects;

/**
 * A group of {@code Matrix} that contain the result of LU-decomposition
 * @param p The permutation matrix
 * @param l The lower triangle matrix
 * @param u The upper triangle matrix
 */
public record LUMatrixGroup(Matrix p, Matrix l, Matrix u) {
}
