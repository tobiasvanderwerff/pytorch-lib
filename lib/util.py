# Taken from https://stackoverflow.com/questions/10724854/how-to-do-a-conditional-decorator-in-python#answer-10724898
def conditional_decorator(dec, condition):
    def decorator(func):
        if condition:
            return dec(func)
        else:
            return func

    return decorator
