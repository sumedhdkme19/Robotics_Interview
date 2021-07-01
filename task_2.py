import cv2
import numpy as np
import os
shapes = {}

def ratio(a,b): #finds the ratio between two numbers a and b and returns a boolean value, whether they are almost equal or not
    result = False
    if a == 0.0 and b == 0.0 :
        result = True
    elif b == 0.0 or a == 0.0 :
        pass
    else :
        r = np.absolute(a/b)
        if (r >= 0.95) and (r <= 1.05) : #tolerence of 0.05 is granted because a and b may not be exactly equal when pixels are taken into consideration
            result = True
        else :
            pass
    return result


def get_coordinates(approx): #this function computes and returns the coordinates of the contour corners
    n = approx.ravel()
    a = [n[0], n[1]]
    b = [n[2], n[3]]
    c = [n[4], n[5]]
    d = [n[6], n[7]]
    return a,b,c,d


def distance(a, b): #this function finds the distance between two point a and b
    dis = np.sqrt(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))
    return dis


def are_all_sides_equal(a,b,c,d) : #checks whether a quadrilateral ABCD has all equal sides
    result = False
    d1, d2, d3, d4 = distance(a,b), distance(b,c), distance(c,d), distance(d,a)
    if ratio(d1,d2) and ratio(d2,d3) and ratio(d3,d4) :
        result = True
    return result


def are_diagonals_equal(a,b,c,d): #checks whether a quadrilateral ABCD has its diagonals equal
    result = False
    d1 = distance(a,c)
    d2 = distance(b,d)
    if ratio(d1,d2):
        result = True
    return result 


def are_opposites_parallel(p,q,r,s): #checks whether a quadrilateral PQRS has its opposite sides parallel
    result = False
    if ((p[0] - q[0]) != 0) :
        m1 = (p[1] - q[1]) / (p[0] - q[0])
    else :
        m1 = 0.0
    if (r[0] - s[0]) != 0 :
        m2 = (r[1] - s[1]) / (r[0] - s[0])
    else :
        m2 = 0.0
    result = ratio(m1,m2)
    return result


def which_quadrilateral(approx): #this function returns the types of the quadrilateral 
    a,b,c,d = get_coordinates(approx)
    r1, r2 = are_opposites_parallel(a,b,c,d), are_opposites_parallel(d,a,c,b)
    if r1 ^ r2 :
        return 'Trapezium'
    elif r1 and r2 :
        r3 = are_all_sides_equal(a,b,c,d)
        if not(r3) :
            return 'Parallelogram'
        else :
            r4 = are_diagonals_equal(a,b,c,d)
            if r4:
                return 'Square'
            else :
                return 'Rhombus'
    else :
        return 'Quadrilateral' 


def shape_detector(noc, approx): #This function detects the shape of the identified contour
    if noc == 3 : #noc stands for "number of corners"
        return 'Triangle'
    elif noc == 4 :
        return which_quadrilateral(approx)
    elif noc == 5 :
        return 'Pentagon'
    elif noc == 6 :
        return 'Hexagon'
    #If the number of corners detected in the contour is more than 6 it has to be a circle. [As ellipse is ruled out]
    else :
        return 'Circle' 


def getcontour(colour, img): #this function gets the contours of the shapes in the image img and compoutes their area, centroid and shape
    d = {}
    ret,thresh = cv2.threshold(img,100,255,cv2.THRESH_BINARY)
    c, h = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in c :
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02*perimeter, True)
        noc = len(approx) #noc stands for number of corners
        x, y, w, h = cv2.boundingRect(approx)
        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        shape = shape_detector(noc, approx)
        area = area * 10
        area = int(area)
        area = area / 10
        d[shape] = [colour, area, cX, cY]   
    return d


def red_coloured_shape_detection(img): #This function detects the shape, area, centroid of all red coloured shapes
    s = {}
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    l_r = np.array([0,70,50])
    u_r = np.array([10, 255, 255])
    mask_red = cv2.inRange(hsv, l_r, u_r)
    s = getcontour('red', mask_red)
    return s #returns a dictionary with the details of all red coloured shapes


def blue_coloured_shape_detection(img):  #This function detects the shape, area, centroid of all blue coloured shapes
    s = {}
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    l_b = np.array([94,80,2])
    u_b = np.array([126,255,255])
    mask_blue = cv2.inRange(hsv, l_b, u_b)
    s = getcontour('blue', mask_blue) 
    return s #returns a dictionary with the details of all blue coloured shapes


def green_coloured_shape_detection(img):  #This function detects the shape, area, centroid of all green shapes
    s = {}
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    l_g = np.array([25, 52, 72])
    u_g = np.array([102, 255, 255])
    mask_green = cv2.inRange(hsv, l_g, u_g)
    s = getcontour('green', mask_green) 
    return s #returns a dictionary with the details of all green coloured shapes




def scan_image(img_file_path):

    global shapes

    img = cv2.imread(img_file_path)
    d1, d2, d3 = {}, {}, {}
    d1 = blue_coloured_shape_detection(img)
    d2 = red_coloured_shape_detection(img)
    d3 = green_coloured_shape_detection(img)
    s = {**d1, **d2, **d3}
    shapes = {k: v for k, v in sorted(s.items(), key=lambda item: item[1][1], reverse = True)}
    return shapes

if __name__ == '__main__':

    curr_dir_path = os.getcwd()
    print('Currently working in '+ curr_dir_path)

    img_dir_path = curr_dir_path
    img_file_path = img_dir_path + 'shapes.jpg'

    print('\n============================================')
    print('\nLooking for shapes' +'.jpg')

    if os.path.exists('shapes.jpg'):
        print('\nFound shapes.png')
    
    else:
        print('\n[ERROR] shapes.png not found. Make sure "Samples" folder has the selected file.')
        exit()
    
    print('\n============================================')

    try:
        print('\nRunning scan_image function with ' + img_file_path + ' as an argument')
        shapes = scan_image(img_file_path)

        if type(shapes) is dict:
            print(shapes)
            print('\nOutput generated. Please verify.')
        
        else:
            print('\n[ERROR] scan_image function returned a ' + str(type(shapes)) + ' instead of a dictionary.\n')
            exit()

    except Exception:
        print('\n[ERROR] scan_image function is throwing an error. Please debug scan_image function')
        exit()

    print('\n============================================')

    choice = input('\nWant to run your script on all the images in Samples folder ? ==>> "y" or "n": ')

    if choice == 'y':

        file_count = 2
        
        for file_num in range(file_count):

            # path to image file
            img_file_path = img_dir_path + 'shapes.jpg'

            print('\n============================================')
            print('\nLooking for Sample' + str(file_num + 1) + '.png')

            if os.path.exists('Samples/Sample' + str(file_num + 1) + '.png'):
                print('\nFound Sample' + str(file_num + 1) + '.png')
            
            else:
                print('\n[ERROR] Sample' + str(file_num + 1) + '.png not found. Make sure "Samples" folder has the selected file.')
                exit()
            
            print('\n============================================')

            try:
                print('\nRunning scan_image function with ' + img_file_path + ' as an argument')
                shapes = scan_image(img_file_path)

                if type(shapes) is dict:
                    print(shapes)
                    print('\nOutput generated. Please verify.')
                
                else:
                    print('\n[ERROR] scan_image function returned a ' + str(type(shapes)) + ' instead of a dictionary.\n')
                    exit()

            except Exception:
                print('\n[ERROR] scan_image function is throwing an error. Please debug scan_image function')
                exit()

            print('\n============================================')

    else:
        print('')
