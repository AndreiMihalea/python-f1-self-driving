import win32api
import win32con

keylist = "\bABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789"

def pressed_keys():
	keys = []
	for key in keylist:
		if win32api.GetAsyncKeyState(ord(key)):
			keys.append(key)
	return keys