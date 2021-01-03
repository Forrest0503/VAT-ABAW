import os
import time
# import face_alignment
from skimage import io
import cv2
import numpy as np

def generate_VA_set_from_cropped_aligned(img_dir, annotation_dir):
    train_ids = []
    val_ids = []
    input_VA_SET = os.path.join('annotations', 'VA_Set')
    output_root = 'dataset'
    output_VA_SET = os.path.join(output_root, 'VA_Set')
    output_train_image = os.path.join(output_VA_SET, 'Training_Set', 'image')
    output_val_image = os.path.join(output_VA_SET, 'Validation_Set', 'image')

    def mkdirs(img_dir, annotation_dir):
        for each in os.listdir(os.path.join(input_VA_SET, 'Training_Set')):
            train_ids.append(each.split('.')[0])
        for each in os.listdir(os.path.join(input_VA_SET, 'Validation_Set')):
            val_ids.append(each.split('.')[0])

        print('train samples: ' + str(len(train_ids)))
        print('val samples: ' + str(len(val_ids)))
        
        # training set label
        os.makedirs(os.path.join(input_VA_SET, 'Training_Set'), exist_ok=True)
        # validation set label
        os.makedirs(os.path.join(input_VA_SET, 'Validation_Set'), exist_ok=True)
        # training set image
        os.makedirs(output_train_image, exist_ok=True)
        # validation set image
        os.makedirs(output_val_image, exist_ok=True)
        return True

    if mkdirs(img_dir, annotation_dir):
        os.system('cp -r ' + os.path.join(input_VA_SET, 'Training_Set') + ' ' + os.path.join(output_VA_SET, 'Training_Set'))
        os.system('mv ' + os.path.join(output_VA_SET, 'Training_Set', 'Training_Set') + ' ' + \
        os.path.join(output_VA_SET, 'Training_Set', 'label'))

        os.system('cp -r ' + os.path.join(input_VA_SET, 'Validation_Set') + ' ' + os.path.join(output_VA_SET, 'Validation_Set'))
        os.system('mv ' + os.path.join(output_VA_SET, 'Validation_Set', 'Validation_Set') + ' ' + \
        os.path.join(output_VA_SET, 'Validation_Set', 'label'))

        for each in train_ids:
            os.system('cp -r ' + os.path.join(img_dir, each) + ' ' + output_train_image)

        for each in val_ids:
            os.system('cp -r ' + os.path.join(img_dir, each) + ' ' + output_val_image)

def generate_VA_submission_set_from_cropped_aligned(img_dir, annotation_dir):
    train_ids = {}
    val_ids = {}
    input_VA_SET = os.path.join('annotations', 'VA_Set')
    output_root = 'dataset'
    output_VA_SET = os.path.join(output_root, 'VA_Set')
    output_test_image = os.path.join(output_VA_SET, 'Testing_Set', 'image')

    def mkdirs(img_dir, annotation_dir):
        for each in os.listdir(os.path.join(input_VA_SET, 'Training_Set')):
            train_ids[each.split('.')[0]] = 1
        for each in os.listdir(os.path.join(input_VA_SET, 'Validation_Set')):
            val_ids[each.split('.')[0]] = 1

        print('train samples: ' + str(len(train_ids)))
        print('val samples: ' + str(len(val_ids)))
        
        # testing set image
        os.makedirs(output_test_image, exist_ok=True)
        return True

    if mkdirs(img_dir, annotation_dir):
        for each in os.listdir(img_dir):
            if each in train_ids or each in val_ids:
                continue
            os.system('cp -r ' + os.path.join(img_dir, each) + ' ' + output_test_image)

def generate_landmarks_for_one_video(video_path, output_root):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    preds = fa.get_landmarks_from_directory(video_path)

    for path, lms in preds.items():
        if not os.path.exists(os.path.join(output_root, 'landmarks', path.split('/')[-2])):
            os.makedirs(os.path.join(output_root, 'landmarks', path.split('/')[-2]))

        out_path = os.path.join(output_root, 'landmarks', os.path.join(path.split('/')[-2], path.split('/')[-1]))
        out_img = np.zeros((112,112), np.uint8)

        if lms == None: # no face
            cv2.imwrite(out_path, out_img)

        else:
            for lm in lms[0]:
                point = ((lm[0], lm[1]))
                cv2.circle(out_img, point, 2, (255,255,255), -1)
            
            # cv2.imshow(out_path, out_img)
            # cv2.waitKey(1000)
            # cv2.destroyAllWindows()
            cv2.imwrite(out_path, out_img)

def generate_landmarks(root):
    videos = os.listdir(os.path.join(root, 'image'))
    for video_id in videos:
        print("processing: ", os.path.join(root, 'image', video_id))
        generate_landmarks_for_one_video(video_path=os.path.join(root, 'image', video_id), \
            output_root=root)

if __name__ == "__main__":
    generate_VA_set_from_cropped_aligned('./cropped_aligned', 'annotations')
    generate_VA_submission_set_from_cropped_aligned('./cropped_aligned', 'annotations')

