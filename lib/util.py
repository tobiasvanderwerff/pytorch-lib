from functools import wraps
from knockknock import email_sender


def conditional_email_sender(func):
    """
    Send an email using knockknock if an email was specified using the email_to_notify
    command line argument.
    """

    @wraps(func)
    def decorator(*args):
        args_ = args[0]
        if args_.email_to_notify is not None:
            return email_sender(recipient_emails=[args_.email_to_notify])(func)(*args)
        else:
            return func(*args)

    return decorator
