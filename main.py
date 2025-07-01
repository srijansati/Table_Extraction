from models import *

dir = 'Dataset'

for file in os.listdir(dir + '/PDF'):
    
    file_path = dir + '/PDF/' + file
    table_dir = dir +'/Retrived_Images/' + file
    detection_dir = dir + '/Detection/'+file

    #creates the output directory if it dosent exist
    if not os.path.exists(table_dir):
        os.mkdir(table_dir)

    #convert every page of pdf to image
    ConvertToImage(pdf_path= file_path, output_path= table_dir)

    for page in os.listdir(dir + '/Retrived_Images/' + file):
        table_structure_recognition(file_path = table_dir + '/' + page, detection_dir = detection_dir, page = page)






