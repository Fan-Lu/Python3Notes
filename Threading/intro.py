"""
Description: Introduction to python3 threading

"""

import threading


def main():
    print(threading.active_count()) # Return the number of Thread objects currently alive.
                                    # count is equal to the length of the list returned by enumerate().
    print(threading.enumerate())    # Return a list of all Thread objects currently alive.
    print(threading.current_thread())   # Return the current Thread object, corresponding to the caller's thread of control.


if __name__ == '__main__':
    main()
