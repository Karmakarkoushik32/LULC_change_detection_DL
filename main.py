from change_detection_pipeline import main

if __name__ == "__main__":
    image_path_1 = r'E:\Drone_tech_lab_assignment\project_change_detection_v2\LULC_change_detection_DL\datasets\raw_images\Phase1.tif'
    image_path_2 = r'E:\Drone_tech_lab_assignment\project_change_detection_v2\LULC_change_detection_DL\datasets\raw_images\Phase2.tif'

    data = main(image_path_1, image_path_2)
    print(data)