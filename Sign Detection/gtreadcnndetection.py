import os
import cv2
def Dataset(Img_dir = 'D:\\ESDproject\\rpntrainimg', ann_dir = 'D:\\ESDproject\\rpntrainann'):
    image_list = []
    bbox_list = []
    for txt_file in os.listdir("D:\\ESDproject\\rpntrainann"):
        bboxes = []
        Image = None
        f = open(r'D:\\ESDproject\\rpntrainann\\gt.txt','r')
        lines = f.readlines()
        for line in lines:
            print(line)
            var = line
            a = var.split(";")
            img = a[0]
            leftCol = a[1]  
            topRow = a[2]
            rightCol = a[3] 
            bottomRow = a[4]
            classid = a[5]
            print(a[0])
            print(a[1])
            print(a[2])
            img_dir = os.path.join('D:/ESDproject/rpntrainimg', img)
            Image = cv2.imread(img_dir, 1)
            print('Reading image file')
            height = len(Image[0])
            width = len(Image[0][0])
                
              
                        # Getting scaled image with 600 as 
                    # length of smaller side                                            
            #if width >= height:
            #    scale = 600.0/height
            #    height= 600
            #    width = scale*width
            #else:
            #    scale = 600.0/width
            #    width = 600
            #    height= scale*width 
                #Image = cv2.resize(Image,(int(width),int(height)))
            print('Getting scaled image')
                    # Getting bounding-boxes 
                    # co-ordinates
            bbox = [int(float(leftCol)),int(float(topRow)),
                                        int(float(rightCol)),int(float(bottomRow))]
            bboxes += [bbox]
            image_list += [Image]
            bbox_list  += [bboxes]
        
    return image_list, bbox_list
    
def Draw_bboxes(bboxes, image):
    
    image = image
    print(bboxes)
    box = bboxes
    print("box is")
    print(box)
    figure = cv2.rectangle(image, (box[0],box[1]), (box[2],box[3]), (0,255,0), 1)
    
    cv2.imshow('Image with Bounding-Boxes', figure)    
    cv2.waitKey(0)   
    cv2.destroyAllWindows()
    
#if __name__=="__main__"

image_list, bbox_list = Dataset()
#print(image_list[0])
print(bbox_list[0][0])
k =0
for image_dummy in image_list:
    Draw_bboxes(bbox_list[0][k],image_dummy);
    k+=1
#cv2.imshow('IMAGE_SHOW', image_list[0])
cv2.waitKey(0)   
# for image in image_list:
    # cv2.imshow('IMAGE_SHOW', image)
    # cv2.waitKey(1000)
    # cv2.destroyAllWindows()
       

        

