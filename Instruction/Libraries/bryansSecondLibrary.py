def factorial(n):
    ''' 
    This function takes an integer input and then prints out the factorial of that number. 
    This function is recursive.
    ''' 
    if n == 1 or n == 0:
        return 1
    else:
        return n * factorial(n-1)