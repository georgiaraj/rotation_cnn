# Example bash script showing how to set up datasets, and train and test the rotation CNN

OUTPUT_FOLDER=~/Testing/interview_data/
MODEL_BASE=$OUTPUT_FOLDER/rot_cnn
TRAIN_DIR=~/Data/VOC2007/JPEGImages #TODO
TEST_DIR=~/Data/reid_data/viper_data/origin
TRAIN_DATASET=$OUTPUT_FOLDER/training_dataset.hdf5
TEST_DATASET=$OUTPUT_FOLDER/test_dataset.hdf5
RESULTS=$OUTPUT_FOLDER/my_results/results.csv

#IMAGE1=<image1.jpg>
#IMAGE2=<image2.jpg>

IM_SIZE=100

# Create data sets
#python3 create_data_set.py  --image_folder $TRAIN_DIR --output_file $TRAIN_DATASET --num_rots 20 --data_im_size $IM_SIZE
#python3 create_data_set.py  --image_folder $TEST_DIR --output_file $TEST_DATASET --data_im_size $IM_SIZE --image_format 'bmp' --save_images

# Train the CNN
python3 train_rot_cnn.py --train_data_file $TRAIN_DATASET --output_model_base $MODEL_BASE --retrain

# Test dataset
python3 test_data_set.py --test_data_file $TEST_DATASET --model_base $MODEL_BASE --results_file $RESULTS

# Test a single image pair
#python3 test_one_pair.py --first_image $IMAGE1 --second_image $IMAGE2 --data_im_size $IM_SIZE --model_base $MODEL_BASE
