package exceptions;

public class NoInverseException extends ArithmeticException{
    public NoInverseException() {
    }

    public NoInverseException(String s) {
        super(s);
    }
}
