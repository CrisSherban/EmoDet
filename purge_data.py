import os


def purge():
    os.system("python manage.py flush --no-input")
    os.system("rm -r main/pics/*")
