''' 

Accretion discs primarily form around compact objects, such as black holes or young stellar objects (YSOs), 
during the process of accretion. These discs result from the infall of matter onto the central object due to gravitational forces. 
On the other hand, protoplanetary discs form around young stars as a byproduct of the star formation process.


Smooth Particle Hydrodynamics (SPH) is a computational method used to simulate fluid dynamics and other physical phenomena. 
It represents a fluid or a continuous medium as a collection of discrete particles, 
where each particle carries certain properties such as mass, position, velocity, and other attributes.


For the SPH simulation, we represent the fluid as a collection of particles. 
Each particle has an initial position, velocity and mass, with both position and velocity being vectors.

'''


# GOVERNING EQUATIONS
'''
https://sedssastrablog.files.wordpress.com/2021/10/image-1.png

The polytropic process equation describes gas expansion and compression, including heat transfer.

'''

import numpy as np
import matplotlib.pyplot as plt

# Constants

N = 4000             # Number of Particles.
t = 0                # start time of simulation.
tEnd = 8            # end time for simulation.
dt = 0.01            # timestep
M = 1.5              # star mass
R = 1.00             # Star Radius
h = 0.1              # smoothing length
k = 0.1              # equation of state constant
n = 1                # Polytropic index
nu = 1               # Damping
m = M/N              # Single-Particle mass
lmbda = 2.01         # lambda for gravity
Nt = int(np.ceil(tEnd/dt))       # Number of time steps.


# The space and time dependent variable called smoothing length (h) is used to determine the region of influence of the neighboring particles.

# Initial Conditions

def initial():
    np.random.seed(41)  #set random number generator seed.
  # We use NumPy random seed when we need to generate pseudo-random numbers in a repeatable way.
    
    pos = np.random.randn(N,2)  # randomly selected positions and velocities.
  
  # np.random.randn(x,y) will create an array of defined shape, filled with random floating-point samples.
  # (x) will create an array of 1D. (x,y) will create an array of 2D.  
    
    vel = np.zeros(pos.shape)

  # np.zeros will create an array with inputs = 0 or 0.. 
  # (pos.shape) here tells that the array of 'vel' should be of same dimensions as 'pos array'.

    return pos, vel


# KERNAL FUNCTION

'''

In SPH, a smoothing function, known as the kernel function, is used to distribute the mass of each particle in space. 
The purpose of the kernel function is to define the influence of a particle on its neighboring particles.
It determines how the properties of a particle, such as density or pressure, affect nearby particles within a specified radius.
The kernel function assigns a weight or contribution to each neighboring particle based on their spatial proximity.


When calculating properties like density or pressure for a particle, the kernel function is applied 
to all neighboring particles within the defined radius.
The contribution of each neighboring particle is weighted based on its distance from the particle of interest.
The closer particles have a higher contribution, while particles farther away have a lesser influence.

The Kernel Function is derived from the Dirac function.
There are many smoothing kernels, like the Gaussian Kernel, Cubic spline Kernel,
Quintic Kernel, etc.

We make use of the Gaussian Kernel.
Gaussian Kernel is a kernel with the shape of a Gaussian (normal distribution) curve.

A Gaussian kernel is a mathematical function that is used to calculate the similarity or weight between points in a dataset. 
It is commonly used in image processing, computer vision, and machine learning tasks such as smoothing, blurring, and feature extraction.

The Gaussian kernel is based on the Gaussian distribution, also known as the bell curve. 
It has a characteristic shape with a peak at the center and gradually decreases towards the edges. 
The kernel assigns higher weights to points that are closer to the center and lower weights to points that are farther away.



'''

# Gaussian-Smoothing Kernel

# https://sedssastrablog.files.wordpress.com/2021/10/image-2.png

def kernel(x,y,h):
    """
    Input:
      x : matrix of x positions
      y : matrix of y positions
      h : smoothing length

    Output:
      w : evaluated smoothing function
    
    """

   # Calculate |r| 

    r = np.sqrt(x**2 + y**2)

   # Calculate the value of smoothing function

    w = (1.0/(h*np.sqrt(np.pi)))**3 * np.exp(-r**2/h**2)

   # Return

    return w


# Smoothing Kernel Gradient

def gradKernel(x, y, h):
    """
    Inputs:
      x : matrix of x positions
      y : matrix of y positions
      h : smoothing length

    Outputs:
      wx, wy : the evaluated gradient
    """

   # Calculate |r|

    r = np.sqrt(x**2 + y**2)

   # Calculate the scalar part of the gradient.

    n = -2 * np.exp(-r**2/ h**2) / h**5 / (np.pi)**(3/2)

   # Vector parts of the gradient

    wx = n * x
    wy = n * y 

   # Return the gradient vector

    return wx, wy


# Density Calculation 

'''
In SPH, the density at a particular point can be found through the formula:

https://sedssastrablog.files.wordpress.com/2021/10/image-3.png

We need to find the distance between all particles to get the density at a particular position.

'''

# Solve for the (r_i - r_j) term in the density formula

def magnitude(ri , rj):
    """
    Inputs:
      ri = M x 2 matrix of positions
      rj = N x 2 matrix of positions

    Output:
      dx, dy : M x N matrix of separations.
    
    """

    M = ri.shape[0]
   
    # [0] is used to access the first element of the tuple returned by 'ri.shape', which represents the number of rows.
    # M = ri.shape[0] assigns the number of rows in the 'ri' array to the variable M. 
   
    N = rj.shape[0]

    # get x, y of r_i

    r_i_x = ri[:,0].reshape((M,1))
    r_i_y = ri[:,1].reshape((M,1))

    '''
    ~ ri is a two dimensional array.
    ~ ri[:,0] selects all the elements from the first column of the 'ri' array.
      The colon ':' represents all the rows, and '0' represents the first column
    ~ .reshape((M,1)) creates a new two dimensional array with 'M' rows and 1 column
    
    In summary, the code takes the first column of the 'ri' array, and then 
    reshapes it into a new array with 'M' rows and 1 column. The resulting array is
    assigned to the variable 'r_i_x'.
    '''
    # get x, y of r_j

    r_j_x = rj[:,0].reshape((N,1))
    r_j_y = rj[:,1].reshape((N,1))

    # get r_i - r_j

    dx = r_i_x - r_j_x.T

    # r_j_x.T is the transpose of the r_j_x array. '.T' attribute is used to obtain the transpose of an array. 
    
    dy = r_i_y - r_j_y.T

    return dx, dy


# Get density at sample points

def density(r, pos, m, h):
    """
    Inputs:
      r   : M x 3 matrix of sampling locations
      pos : N x 3 matrix of particle positions
      m   : particle mass
      h   : smoothing length

    Output:
      rho : M x 1 vector of densities
    
    """

    M = r.shape[0]
    dx, dy = magnitude(r, pos);
    rho = np.sum(m * kernel(dx, dy, h), 1).reshape((M,1))

# 1 in np.sum(..., 1) specifies the axis along which the sum should be performed. In this case, it sums the elements along axis 1.
# In summary, the code calculates the sum of the element-wise multiplication of 'm' with the result of the kernel function applied to 'dx' and 'dy', along axis 1. 
# The resulting array is then reshaped into a new array with M rows and 1 column, which is assigned to the variable 'rho'.
    
    return rho


# The code dx, dy = magnitude(r, pos) performs tuple unpacking, which means that 
# it assigns the first value of the tuple returned by 'magnitude' to the variable dx, and the second value to the variable dy.

# dx and dy are being assigned the values returned by the 'magnitude' function when 'r' and 'pos' are passed as arguments.

# Pressure Calculation

"""
Since we already have the ρ values from the previous step, we can directly calculate the pressure at each point using the formula:

https://sedssastrablog.files.wordpress.com/2021/10/image-4.png

"""

def pressure(rho, k, n):
    """
    Inputs:
    rho : vector of densities
    k   : equation of state constant
    n   : polytropic index

    The polytropic index (n) is a measure of the work done by the system. 

    Output:
    P : pressure
    
    """

    P = k * rho**(1+1/n)

    return P

# Acceleration Constant
'''
In order to track the motion of the particles, we need to know their position at all time steps.
But in order to know their position, we need to know the velocity.
And to know the velocity, we require the acceleration of each particle.

We can calculate the acceleration acting on each particle at every time step using the following discretization formula:

https://sedssastrablog.files.wordpress.com/2021/10/image-5.png


'''
    
# Calculate Acceleration on Particles

def acceleration(pos, vel, m, h, k, n, lmbda, nu):
    """
    Inputs: 

    pos    : N x 2 matrix of positions
    vel    : N x 2 matrix of velocities
    m      : particle mas
    h      : smoothing length
    k      : equation of state constant
    n      : polytropic index
    lmbda  : external force constant
    nu     : viscosity

    Output:

    a : N x 2 matrix of accelerations.

    """


    N = pos.shape[0]

    # calculate densities
    rho = density(pos,pos,m,h)

    # Get Pressures
    P = pressure(rho,k,n)

    # Get pairwise distances and gradients
    dx, dy = magnitude(pos, pos)
    dWx, dWy = gradKernel(dx, dy, h)

    # Add Pressure Contribution to Acceleration

    ax = - np.sum(m * (P/rho**2 + P.T/rho.T**2) * dWx, 1).reshape((N,1))
    ay = - np.sum(m * (P/rho**2 + P.T/rho.T**2) * dWy, 1).reshape((N,1))

    # Pack Acceleration Components
    a = np.hstack((ax,ay))

    '''
    a = np.hstack((ax,ay)) horizontally stacks ax and ay arrays together, resulting in a new array 'a' where 
    the elements from 'ax' are placed to the left of the elements from 'ay'. 
    The shape of the resulting array 'a' will depend on the shapes of 'ax' and 'ay', but the number of rows will remain the same, 
    and the number of columns will be the sum of the columns in ax and ay.
    '''

    # Add External Forces
    a += -lmbda * pos - nu * vel

    # Return total acceleration
    return a


# SIMULATION
'''
The setup has been completed. The simulation Requires two main parts.

Firstly, we need to set the main loop to run the simulation forward in time.
Next, we need to stitch the frames together to form an animation
'''

# MAIN LOOP
'''
Here, we set up the loop that controls the simulation in time domain.
Here, we update the velocitty and position using Newton's laws of motion:

https://sedssastrablog.files.wordpress.com/2021/10/image-6.png

'''

import os 
'''
By importing the os module, you gain access to its functions and methods, 
allowing you to perform operations related to the operating system within your Python program.
'''

import glob 
''' 
The glob module allows you to find files and directories that match a specified pattern, 
similar to how you might use wildcards in a command-line interface. It is often used for tasks such as 
file searching, file listing, and retrieving file paths based on specific criteria. 
'''

import tqdm
'''
The tqdm module provides a simple and customizable way to visualize the progress of a loop or an iterable. 
It creates a progress bar that shows the elapsed time, estimated time remaining, and the percentage of completion.
'''


# Creates folder if it doesn't exist

if not os.path.exists('output'):   # If the folder 'output' doesn't exist, with the help of 'os' module, we can make one.
    os.mkdir('output') # Will make a directory(folder) named 'Output'.


# Else, deletes all images inside it if the folder exists already.

else:
    files = glob.glob('output/*.png')  # 'Glob' searches for the file.Once it finds them, 'os' is used to remove the directory
    for f in files:
        os.remove(f)


# Disable inline printing to prevent all graphs from being shown.
pos, vel = initial()

# Start LOOP

for i in tqdm.tqdm(range(Nt)):
    # Update values
    acc = acceleration(pos, vel, m, h, k, n, lmbda, nu)
    vel += acc * dt
    pos += vel * dt
    rho = density(pos, pos, m, h)

    # PLOT
    fig,ax = plt.subplots(figsize = (6,6))

    plt.sca(ax)
    plt.cla()

    # Get colour from density
    cval = np.minimum((rho - 3)/3, 1).flatten()

    '''
     The code subtracts 3 from each element of the array rho, then divides the result by 3. The resulting array 
     is compared element-wise with 1 using np.minimum(), which takes the minimum value between each element of the resulting array and 1. 
     Finally, the resulting array is flattened into a one-dimensional array and assigned to the variable cval.
     '''

    # Place particles on map with colors
    plt.scatter(pos[:,0], pos[:,1], c=cval, cmap = plt.cm.viridis, s = 2, alpha = 0.75)

    '''
    The given code utilizes the plt.scatter() function from the matplotlib.pyplot module to create a scatter plot. 
    Here's a breakdown of each component:

   ~ pos[:,0] and pos[:,1] are used to extract the first and second columns of the pos array, respectively. 
     These columns represent the x and y coordinates of the data points.
   ~ 'c = cval' assigns the values of the cval array to the c parameter of the plt.scatter() function. 
     This parameter determines the color of each data point based on the values in cval.
   ~ 'cmap = plt.cm.autumn' specifies the color map to be used for mapping the c values to colors. 
     In this case, the 'autumn' color map is used.
   ~ s=5 sets the size of the markers in the scatter plot to 5 pixels. 
   ~ alpha=0.75 sets the transparency of the markers to 0.75, making them partially transparent.

     In summary, the code creates a scatter plot where each data point is represented by a marker. 
     The x and y coordinates of the data points are taken from the 'pos' array. 
     The color of each marker is determined by the values in the 'cval' array, using the 'autumn' color map. 
     The 'size' of the markers is set to 5 pixels, and their 'transparency' is set to 0.75.
    
    '''

    '''
    # Here are some commonly used color maps:

- `plt.cm.viridis`: A color map that goes from deep blue to vibrant yellow.
- `plt.cm.jet`: A color map with a full spectrum of colors, ranging from blue to red.
- `plt.cm.coolwarm`: A color map that transitions smoothly between cool colors (blues) and warm colors (reds).
- `plt.cm.magma`: A color map with a black-purple-red-yellow color progression.
- `plt.cm.plasma`: A color map with a vibrant purple-pink-yellow color progression.
- `plt.cm.inferno`: A color map with a black-purple-orange-yellow color progression.
- `plt.cm.cividis`: A color map designed to be easily distinguishable by individuals with color vision impairment.

 
    '''
    # Set Plot limits and stuff
    ax.set(xlim = (-3.0, 3.0), ylim = (-3.0, 3.0))
    ax.set_aspect('equal', 'box')
    '''
    This code is commonly used when you want to maintain the proportional representation of data in the plot, 
    ensuring that the x-axis and y-axis scales are visually comparable and not distorted.
    '''
    
    ax.set_facecolor('black')
    ax.set_facecolor((.1,.1,.1))

    # 'ax.set_facecolor()' is a method provided by the Axes object in matplotlib to set the background color.
    # (0.1, 0.1, 0.1) is a tuple of RGB values representing a dark gray color.
    
    # Save Plot
    plt.savefig(f'output/{i}.png')
    plt.close()


# ANIMATION
'''
All the frames are ready. we will stitch them together into a video
using OpenCV, and download the simulation.
'''

import cv2

'''
The cv2 module in Python refers to the OpenCV (Open Source Computer Vision) library, which is a popular computer vision library 
used for various image and video processing tasks. The cv2 module provides a Python interface to the functionalities of the OpenCV library.
'''

img_array = []

'''
By assigning an empty list to the variable img_array, you are creating a container that can later be populated with elements. 
This list can be used to store and manipulate data, such as image data in this context.
'''

print("Reading Frames")
# List image paths
imgs_list = glob.glob('output/*.png')
lsorted = sorted(imgs_list, key=lambda x: int(os.path.splitext(x[7:])[0]))


for filename in tqdm.tqdm(lsorted):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

'''
Here's an explanation of each step:

1. imgs_list = glob.glob('output/*.png'): 

The 'glob' module is imported, and the 'glob.glob()' function is used to search for files with the extension .png in the output directory. 
The resulting list of file paths is assigned to the 'imgs_list' variable

2. lsorted = sorted(imgs_list, key=lambda x: int(os.path.splitext(x[7:])[0])): 

'The sorted()' function is used to sort the 'imgs_list' in ascending order based on the numeric portion of the file names. 
This is achieved by providing a key function that extracts the numeric part from each file name using 'os.path.splitext()' to split 
the file name and extension, and then converting it to an integer. The sorted list is assigned to the 'lsorted' variable.

3. for 'filename' in tqdm.tqdm(lsorted): : 

A loop is initiated that iterates over each file name in the sorted list 'lsorted'. The 'tqdm.tqdm()' function is used to wrap 'lsorted', 
providing a progress bar for the loop.

4. img = cv2.imread(filename): 

For each iteration of the loop, the 'cv2.imread()' function from the OpenCV (cv2) module is used to read the image data 
from the file specified by filename. The resulting image is assigned to the img variable.

5. height, width, layers = img.shape: 

The shape 'attribute' of the img array is accessed to retrieve the height, width, and number of color channels (layers) of the image. 
These values are assigned to the height, width, and layers variables, respectively.

6. size = (width,height): 

The width and height values are used to create a tuple size representing the dimensions of the image.

7. img_array.append(img): 

The 'img array', representing the image data read from the file, is appended to the 'img_array' list.

The code reads a list of image file paths matching a specific pattern using the glob module, sorts them based on the 
numeric part of the file names, and then iterates over the sorted list. For each file, it reads the image data using OpenCV, 
extracts the image dimensions, and appends the image data to the img_array list. The tqdm module is used to provide a progress bar 
for the loop, showing the progress of reading and processing the images.


'''

out = cv2.VideoWriter('simulation.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 60, size)

# Write to video
print("Writing frames")
for i in tqdm.tqdm(range(len(img_array))):
    out.write(img_array[i])
out.release()


'''
~ cv2.VideoWriter_fourcc(*'MP4V'): 

This parameter specifies the four-character code (FourCC) representing the video codec used for compression. 
In this example, the codec used is MP4V, which corresponds to the 'MPEG-4' Visual codec. 
The 'cv2.VideoWriter_fourcc()' function is used to convert the 'FourCC' code to the required format.

~ 60: 

This parameter specifies the frames per second (fps) of the output video. In this case, the video will have a frame rate of 60 frames 
per second. You can modify this value to set the desired frame rate.

~ The out object returned by 'cv2.VideoWriter' represents the video writer, which can be used to write individual frames to the video file.

~ After creating the 'out' object, you can proceed to write 'frames' to the video file using the 'write()' method of the 'cv2.VideoWriter' object. 
Each frame should be in the correct format and size specified during the video writer initialization.

~ Once all the frames have been written, remember to 'release' the video writer using the 'release()' method 
to close the file and finalize the video creation.

'''





