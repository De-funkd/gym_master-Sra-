- observation space - 
	- Greyscale - it is a 255 * 160 matrix which stores a value of the intensity of the pixel from 0 to 255 as a 8 bit unsigned integer .
	- Ram - it is a simple 1 dimensional vector with 128 values in it representation the internal states of the game like ball position , paddle position etc . 

- action spaces - 6 action spaces but only 2 of them are actually useful which is the left and right paddle ie 2,3 action space .

- reward - u get a reward if the ball crosses opponents paddle . 