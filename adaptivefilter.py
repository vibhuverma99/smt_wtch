import numpy as np
import matplotlib.pyplot as plt

def checkDesiredSignal(d, N, M):
    if len(d) < N+M-1:
        raise ValueError('Desired signal must be >= N+M-1 or len(u)')


def checkNumTaps(M):
    if type(M) is not int:
        raise TypeError('Number of filter taps must be type integer')
    elif M <= 0:
        raise ValueError('Number of filter taps must be greater than 0')


def checkInitCoeffs(c, M):
    if len(c) != M:
        err = 'Length of initial filter coefficients must match filter order'
        raise ValueError(err)


def checkIter(N, maxlen):
    if type(N) is not int:
        raise TypeError('Number of iterations must be type integer')
    elif N > maxlen:
        raise ValueError('Number of iterations must not exceed len(u)-M+1')
    elif N <= 0:
        err = 'Number of iterations must be larger than zero, please increase\
 number of iterations N or length of input u'
        raise ValueError(err)


def checkStep(step):
    if type(step) is not float and type(step) is not int:
        raise TypeError('Step must be type float (or integer)')
    elif step < 0:
        raise ValueError('Step size must non-negative')


def checkLeakage(leak):
    if type(leak) is not float and type(leak) is not int:
        raise TypeError('Leakage must be type float (or integer)')
    elif leak > 1 or leak < 0:
        raise ValueError('0 <= Leakage <= 1 must be satisfied')


def checkRegFactor(eps):
    if type(eps) is not float and type(eps) is not int:
        err = 'Regularization (eps) must be type float (or integer)'
        raise ValueError(err)
    elif eps < 0:
        raise ValueError('Regularization (eps) must non-negative')


def checkProjectOrder(K):
    if type(K) is not int:
        raise TypeError('Projection order must be type integer')
    elif (K <= 0):
        raise ValueError('Projection order must be larger than zero')


def lms(u, d, M, step, leak=0, initCoeffs=None, N=None, returnCoeffs=False):
    """
    Perform least-mean-squares (LMS) adaptive filtering on u to minimize error
    given by e=d-y, where y is the output of the adaptive filter.
    Parameters
    ----------
    u : array-like
        One-dimensional filter input.
    d : array-like
        One-dimensional desired signal, i.e., the output of the unknown FIR
        system which the adaptive filter should identify. Must have length >=
        len(u), or N+M-1 if number of iterations are limited (via the N
        parameter).
    M : int
        Desired number of filter taps (desired filter order + 1), must be
        non-negative.
    step : float
        Step size of the algorithm, must be non-negative. Also called Mu.
    Optional Parameters
    -------------------
    leak : float
        Leakage factor, must be equal to or greater than zero and smaller than
        one. When greater than zero a leaky LMS filter is used. Defaults to 0,
        i.e., no leakage.
    initCoeffs : array-like
        Initial filter coefficients to use. Should match desired number of
        filter taps, defaults to zeros.
    N : int
        Number of iterations to run. Must be less than or equal to len(u)-M+1.
        Defaults to len(u)-M+1.
    returnCoeffs : boolean
        If true, will return all filter coefficients for every iteration in an
        N x M matrix. Does not include the initial coefficients. If false, only
        the latest coefficients in a vector of length M is returned. Defaults
        to false.
    Returns
    -------
    y : numpy.array
        Output values of LMS filter, array of length N.
    e : numpy.array
        Error signal, i.e, d-y. Array of length N.
    w : numpy.array
        Final filter coefficients in array of length M if returnCoeffs is
        False. NxM array containing all filter coefficients for all iterations
        otherwise.
    Raises
    ------
    TypeError
        If number of filter taps M is not type integer, number of iterations N
        is not type integer, or leakage leak is not type float/int.
    ValueError
        If number of iterations N is greater than len(u)-M, number of filter
        taps M is negative, or if step-size or leakage is outside specified
        range.
    Minimal Working Example
    -----------------------
    >>> import numpy as np
    >>>
    >>> np.random.seed(1337)
    >>> ulen = 2000
    >>> coeff = np.concatenate(([1], np.zeros(10), [-0.9], np.zeros(7), [0.1]))
    >>> u = np.random.randn(ulen)
    >>> d = np.convolve(u, coeff)
    >>>
    >>> M = 20  # No. of taps
    >>> step = 0.03  # Step size
    >>> y, e, w = lms(u, d, M, step)
    >>> print np.allclose(w, coeff)
    True
    Extended Example
    ----------------
    >>> import numpy as np
    >>>
    >>> np.random.seed(1337)
    >>> N = 1000
    >>> coeffs = np.concatenate(([-3], np.zeros(9), [6.9], np.zeros(8), [0.7]))
    >>> u = np.random.randn(20000)  # Note len(u) >> N but we limit iterations
    >>> d = np.convolve(u, coeffs)
    >>>
    >>> M = 20  # No. of taps
    >>> step = 0.02  # Step size
    >>> y, e, w = lms(u, d, M, step, N=N, returnCoeffs=True)
    >>> y.shape == (N,)
    True
    >>> e.shape == (N,)
    True
    >>> w.shape == (N, M)
    True
    >>> # Calculate mean square weight error
    >>> mswe = np.mean((w - coeffs)**2, axis=1)
    >>> # Should never increase so diff should above be > 0
    >>> diff = np.diff(mswe)
    >>> (diff <= 1e-10).all()
    True
    """
    # Num taps check
    checkNumTaps(M)
    # Max iteration check
    if N is None:
        N = len(u)-M+1
    checkIter(N, len(u)-M+1)
    # Check len(d)
    checkDesiredSignal(d, N, M)
    # Step check
    checkStep(step)
    # Leakage check
    checkLeakage(leak)
    # Init. coeffs check
    if initCoeffs is None:
        initCoeffs = np.zeros(M)
    else:
        checkInitCoeffs(initCoeffs, M)

    # Initialization
    y = np.zeros(N)  # Filter output
    e = np.zeros(N)  # Error signal
    w = initCoeffs  # Initial filter coeffs
    leakstep = (1 - step*leak)
    if returnCoeffs:
        W = np.zeros((N, M))  # Matrix to hold coeffs for each iteration

    # Perform filtering
    for n in range(N):
        x = np.flipud(u[n:n+M])  # Slice to get view of M latest datapoints
        y[n] = np.dot(x, w)
        e[n] = d[n+M-1] - y[n]

        w = leakstep * w + step * x * e[n]
        y[n] = np.dot(x, w)
        if returnCoeffs:
            W[n] = w

    if returnCoeffs:
        w = W
        return y, e, w

if __name__ == '__main__':
    x = np.linspace(-4 , 4 , 100)
    n1 = np.random.normal(0,1,100)
    sig = np.sin(x)+n1
    n2 = np.random.normal(0,1,100)
    M = 6 #desired no. of filter taps i.e length of filter
    initialCoeff = np.zeros(M)

    y,e,w = lms(n1,sig,M,0.01,0,initialCoeff,75,True)
    
    plt.subplot(3,2,1)
    plt.plot(x,np.sin(x))
    plt.xlabel('TIME')
    plt.ylabel('AMP')
    plt.title('SIGNAL WITHOUT NOISE')

    plt.subplot(3,2,5)
    plt.plot(x,sig)
    plt.xlabel('TIME')
    plt.ylabel('AMP')
    plt.title('SIGNAL WITH ADDED NOISE')

    plt.subplot(3,2,3)
    plt.plot(x,n1)
    plt.xlabel('TIME')
    plt.ylabel('AMP')
    plt.title('NOISE N1')

    plt.subplot(3,2,4)
    plt.plot(x,n2)
    plt.xlabel('TIME')
    plt.ylabel('AMP')
    plt.title('NOISE N2')

    plt.subplot(3,2,6)
    plt.plot(x[0:len(y):1],e)
    plt.xlabel('TIME')
    plt.ylabel('AMP')
    plt.title('SIGNAL FILTERED')

    plt.subplot(3,2,2)
    plt.plot(x[0:len(y):1],y)
    plt.xlabel('TIME')
    plt.ylabel('AMP')
    plt.title('SIGNAL Y')

    plt.show()
