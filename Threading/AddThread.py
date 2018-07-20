"""
Description: Add threads
"""

import threading

def thread_job():
    print("This is a thread of %s" % threading.current_thread())

def main():
    print('Adding a thread')
    thread1 = threading.Thread(target=thread_job())
    thread2 = threading.Thread(target=thread_job())
    thread1.start()
    thread2.start()


if __name__ == '__main__':
    main()