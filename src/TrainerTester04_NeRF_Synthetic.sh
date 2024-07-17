#!/bin/bash
# Shree KRISHNAya Namaha

scene_names=(
        "chair"
        "drums"
        "ficus"
        "hotdog"
        "lego"
        "materials"
        "mic"
        "ship"
        )

# parse the input arguments
configs_path=$(realpath $1)
for ARGUMENT in "${@:2}"  # parse from the second argument
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

# Check if the mandatory arguments are provided
if [ -z "$train_set_num" ]; then
    echo "Please provide train_set_num"
    exit 1
fi

# Get root directory path
root_dirpath="$(dirname "$(dirname "$(dirname "$(dirname $configs_path)")")")"

# Get train_num and train_dirpath
train_dirpath=$(dirname $configs_path)
train_num=$(echo $(basename $train_dirpath) | cut -c 6-9)

# Get test_num and test_dirpath
if [ -z "$test_num" ]; then
    # If test_num is not provided, set test_num to train_num
    test_num=$train_num
else
    # If test_num is provided, format it to 4 digits by padding leading zeros
    test_num=$(printf "%04d" $test_num)
fi
test_dirpath=$root_dirpath/runs/testing/test$test_num

# Get train and test set num
train_set_num=$(printf "%02d" $train_set_num)  # format to 2 digits by padding leading zeros
if [ -z "$test_set_num" ]; then
    # If test_set_num is not provided, set test_set_num to train_set_num
    test_set_num=$train_set_num
else
    # If test_set_num is provided, format it to 2 digits by padding leading zeros
    test_set_num=$(printf "%02d" $test_set_num)
fi

# Get conda environment name for creating videos
if [ -z "$video_env_name" ]; then
    video_env_name="SimpleZipNeRF"
fi

# Get framerate for creating videos
if [ -z "$framerate" ]; then
    framerate=30
fi

# Get conda environment name for running QA
if [ -z "$qa_env_name" ]; then
    qa_env_name="SimpleZipNeRF"
fi

# Get gt depth test num
if [ -z "$gt_depth_test_num" ]; then
    gt_depth_test_num="3001"
else
    gt_depth_test_num=$(printf "%04d" $gt_depth_test_num)
fi

# Get qa_masks_dirname
if [ -z "$qa_masks_dirname" ]; then
    qa_masks_dirname="VM$test_set_num"
fi

# Get batch size and render chunk size
if [ -z "$batch_size" ]; then
    batch_size=8192
fi
if [ -z "$render_chunk_size" ]; then
    render_chunk_size=8192
fi

database_dirpath=$root_dirpath/data/databases/NeRF_Synthetic/data
gt_depth_dirpath=$root_dirpath/data/dense_input_radiance_fields/ZipNeRF/runs/testing/test$gt_depth_test_num
test_frames_datapath=$database_dirpath/train_test_sets/set$test_set_num/TestVideosData.csv
qa_filepath=$root_dirpath/src/qa/00_Common/src/AllMetrics04_NeRF_Synthetic.py

# print all variable values
declare -p scene_names configs_path root_dirpath database_dirpath gt_depth_test_num gt_depth_dirpath train_set_num test_set_num test_frames_datapath train_num train_dirpath test_num test_dirpath video_env_name framerate qa_env_name qa_masks_dirname qa_filepath batch_size render_chunk_size

# Sleep for 5 seconds before starting to executing the train/eval/render/qa pipeline
sleep 5s

# Loop over every scene and run training and rendering
for scene_name in ${scene_names[@]};
do
    scene_train_configs_path="$train_dirpath/$scene_name/$scene_name.gin"
    scene_test_configs_path="$test_dirpath/$scene_name/$scene_name.gin"

    # Create the train configs file for the scene
    python utils/ConfigsCreator04_NeRF_Synthetic.py --configs-path $1 --scene-names $scene_name --mode train  --train-set-num $train_set_num

    # Call training
    python train.py --gin_configs=$scene_train_configs_path --gin_bindings="Config.batch_size = $batch_size" --gin_bindings="Config.render_chunk_size = $render_chunk_size"

    # Create the test configs file for the scene
    python utils/ConfigsCreator04_NeRF_Synthetic.py --configs-path $1 --scene-names $scene_name --mode test  --test-num $test_num --test-set-num $test_set_num --train-set-num $train_set_num

    # Call validation
    python eval.py --gin_configs=$scene_test_configs_path --gin_bindings="Config.batch_size = $batch_size" --gin_bindings="Config.render_chunk_size = $render_chunk_size"

    # Run QA
    conda run --live-stream -n $qa_env_name bash -c "python $qa_filepath --demo_function_name demo2 --pred_videos_dirpath $test_dirpath --database_dirpath $database_dirpath --gt_depth_dirpath $gt_depth_dirpath --frames_datapath $test_frames_datapath --pred_frames_dirname predicted_frames --pred_depths_dirname predicted_depths --mask_folder_name $qa_masks_dirname"

    # Create videos
    conda run --live-stream -n $video_env_name bash -c "python utils/VideosCreator04_NeRF_Synthetic.py --test-dirpath $test_dirpath --scene-names $scene_name --framerate $framerate"
done
