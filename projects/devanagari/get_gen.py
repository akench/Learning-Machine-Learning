import pexpect
import getpass
import os
import time

username = input('Enter username: ')
pswd = getpass.getpass('Password: ')

# os.system('sftp {}@lisp.cs.rutgers.edu'.format(username))
# time.sleep(1)
# os.system(pswd)
# time.sleep(1)
# os.system('lcd ~/repos/Learning-Machine-Learning/projects/devanagari/out/')
# time.sleep(.5)
# os.system('cd repos/Learning-Machine-Learning/projects/devanagari/out/')
# time.sleep(.5)
# print('getting files..')
# os.system('get *')
# time.sleep(.5)
# os.system('exit')

def decode(bytes):
	return bytes.decode('utf-8')

child = pexpect.spawn('sftp {}@lisp.cs.rutgers.edu'.format(username))
child.expect('Password')
child.sendline(pswd)
print(decode(child.after))

child.expect('Connected.*')
print(decode(child.after))
child.sendline('lcd ~/repos/Learning-Machine-Learning/projects/devanagari/out/')
# child.expect('sftp>.*')
child.sendline('lls')
child.expect('sftp>.*')
print(decode(child.after))
child.sendline('cd Learning-Machine-Learning/projects/devanagari/out/')
child.expect('sftp>.*')
child.sendline('ls')
child.expect('sftp>.*')
print('getting files.....')
child.sendline('get *');
print(decode(child.after))

print('done! :)')
